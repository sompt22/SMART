from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os

import torch
import torch.utils.data
from opts import opts
from model.model import create_model, load_model, save_model
from model.data_parallel import DataParallel
from logger import Logger
from dataset.dataset_factory import get_dataset
from trainer import Trainer


def set_requires_grad(nets, requires_grad=False):
  """Helper function to set requires_grad for components in a model."""
  if not isinstance(nets, list):
    nets = [nets]
  for net in nets:
    if net is not None:
      for param in net.parameters():
        param.requires_grad = requires_grad

def set_bn_eval(module):
    """Recursively set batch normalization layers to evaluation mode."""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()
    for child in module.children():
        set_bn_eval(child)
        
def initialize_and_freeze_model_components_withbn(opt, model):
    # Freezing base network (if considered as backbone)
    if 'base' in opt.freeze_components and opt.freeze_components['base']:
        set_requires_grad(model.base, requires_grad=False)
    # Freezing task-specific heads
    for head_name in ['hm', 'reg', 'wh', 'embedding', 'tracking', 'ltrb_amodal']:
        if head_name in opt.freeze_components and opt.freeze_components[head_name]:
            set_requires_grad(getattr(model, head_name), requires_grad=False)
    # Example: Freezing additional components like dla_up or ida_up if needed
    # Adjust based on your freeze_components options
    if 'dla_up' in opt.freeze_components and opt.freeze_components['dla_up']:
        set_requires_grad(model.dla_up, requires_grad=False)
    if 'ida_up' in opt.freeze_components and opt.freeze_components['ida_up']:
        set_requires_grad(model.ida_up, requires_grad=False)

    return model        

def initialize_and_freeze_model_components(opt, model):
    # Assuming the model is in training mode by default: model.train()
    
    # Freezing base network (if considered as backbone)
    if 'base' in opt.freeze_components and opt.freeze_components['base']:
        set_requires_grad(model.base, requires_grad=False)
        set_bn_eval(model.base)  # Optionally freeze BN stats in this part
        print("Base model frozen! \n")
        
    # Freezing task-specific heads and other components
    for component_name in ['hm', 'reg', 'wh', 'embedding', 'tracking', 'ltrb_amodal', 'dla_up', 'ida_up']:
        if component_name in opt.freeze_components and opt.freeze_components[component_name]:
            component = getattr(model, component_name, None)
            if component:
                set_requires_grad(component, requires_grad=False)
                set_bn_eval(component)  # Optionally freeze BN stats in these parts
                print(f"{component} frozen! \n")
                
    return model

def get_optimizer(opt, parameters):
  if opt.optim == 'adam':
    print('Using Adam')
    optimizer = torch.optim.Adam(parameters, opt.lr)  # Use 'parameters' directly
  elif opt.optim == 'sgd':
    print('Using SGD')
    optimizer = torch.optim.SGD(parameters, opt.lr, momentum=0.9, weight_decay=0.0001)
  elif opt.optim == 'adamw':
    print('Using AdamW')
    optimizer = torch.optim.AdamW(parameters, opt.lr)
  else:
    assert 0, opt.optim
  return optimizer

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  if not opt.not_set_cuda_env:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  logger = Logger(opt)
  print(f'Unique track ids: {opt.nID}')
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
  model = initialize_and_freeze_model_components(opt, model)
  #print(model)
  #optimizer = get_optimizer(opt, model)
  optimizer = get_optimizer(opt, filter(lambda p: p.requires_grad, model.parameters()))
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, opt, optimizer)

  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
  
  if opt.val_intervals < opt.num_epochs or opt.test:
    print('Setting up validation data...')
    val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), batch_size=1, shuffle=False, num_workers=1,
      pin_memory=True)

    if opt.test:
      _, preds = trainer.val(0, val_loader)
      val_loader.dataset.run_eval(preds, opt.save_dir)
      return
    print(f'Unique track ids after val load: {opt.nID}')

  print('Setting up train data...')
  print("Data is shuffling?:", opt.noshuffle)
  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), batch_size=opt.batch_size, shuffle=opt.noshuffle,
      num_workers=opt.num_workers, pin_memory=True, drop_last=True
  )
  print(f'Unique track ids after train load: {opt.nID}')

  print('Starting training...')
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    logger.write('\n')
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
        if opt.eval_val:
          val_loader.dataset.run_eval(preds, opt.save_dir)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      logger.write('\n')
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)

    if epoch in opt.save_point:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
    if epoch in opt.lr_step:
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
