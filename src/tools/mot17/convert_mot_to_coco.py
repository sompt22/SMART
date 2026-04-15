from pathlib import Path
import runpy


if __name__ == '__main__':
    script_path = Path(__file__).resolve().parents[1] / 'convert_mot_to_coco.py'
    print('Deprecated entrypoint. Running official converter:', script_path)
    runpy.run_path(str(script_path), run_name='__main__')
