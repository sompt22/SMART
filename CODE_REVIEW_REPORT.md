# SMART (Stick via Motion and Recognition Tracker) - Kapsamlı Kod İnceleme Raporu

**Tarih:** 2026-03-18
**İncelenen Modüller:** Tracker, Matching, Model, Training, Detector, Dataset, Kalman Filter, Decode, Post-Process
**Toplam Kaynak Dosya:** ~128 dosya (Python, C++, CUDA, Cython)

---

## 1. TASARIM GEREKLERİ ANALİZİ (Multi-Object Tracking)

### 1.1 Pipeline Bütünlüğü

MOT pipeline'ı **yapısal olarak tamdır** ve şu adımları içerir:

| Adım | Durum | Dosya |
|------|-------|-------|
| Detection (CenterNet-tabanlı) | ✅ Mevcut | `detector.py`, `decode.py` |
| Kalman Filter Prediction | ✅ Mevcut | `kalman_filter.py` |
| 1. Association (Embedding + Motion Gating) | ⚠️ Kritik Bug | `multitracker.py:296`, `matching.py` |
| 2. Association (IoU fallback) | ✅ Mevcut | `multitracker.py:312-331` |
| Unconfirmed Track Handling | ✅ Mevcut | `multitracker.py:334-343` |
| Track Initialization | ✅ Mevcut | `multitracker.py:346-351` |
| Track Lifecycle (Lost/Removed) | ⚠️ Bellek Sızıntısı | `multitracker.py:366` |
| Knowledge Distillation | ⚠️ Kısmen Uygulanmış | `trainer.py`, `losses.py` |
| Multi-Task Learning (Det+Emb+Disp) | ✅ Mevcut | `trainer.py:118-120` |
| ReID Feature Extraction | ✅ Mevcut | `generic_network.py`, `base_model.py` |

### 1.2 Desteklenen Dataset'ler
MOT17, MOT20, DIVOTrack, SOMPT22, CrowdHuman, COCO — tamamı `dataset_factory.py` ile entegre.

### 1.3 Sonuç
Pipeline mimari olarak eksiksizdir ancak **kritik runtime bugları** nedeniyle doğru çalışması mümkün değildir (bkz. Bölüm 2).

---

## 2. KRİTİK HATALAR (Runtime / Compile-time)

### 🔴 SEVİYE: KRİTİK

#### BUG-01: `embedding_distance` Argüman Sırası Ters (Sessiz Yanlış Eşleştirme)
- **Dosya:** `multitracker.py:296` → `matching.py:52`
- **Sorun:** `multitracker.py` fonksiyonu `(strack_pool, detections)` sırasıyla çağırıyor, ama `matching.py` imzası `(detections, tracks)` şeklinde. Maliyet matrisi **transpoze** oluyor (M×N yerine N×M).
- **Etki:** Tüm 1. aşama association yanlış track-detection eşleştirmeleri üretiyor. **Tracking kalitesini doğrudan ve sessizce bozar.**

#### BUG-02: `BaseModel.forward()` — `self.opt` Hiç Atanmamış
- **Dosya:** `base_model.py:67`
- **Sorun:** `__init__` metodu `self.opt = opt` atamasını yapmıyor ama `forward()` metodu `self.opt.model_output_list` erişiyor.
- **Etki:** DLA, ResNet gibi `BaseModel`'den türeyen tüm modellerin ilk forward pass'inde `AttributeError` fırlatır. **Model hiç çalışmaz** (eğer subclass kendisi atamıyorsa).

#### BUG-03: `np.float` Kaldırılmış (NumPy ≥1.24)
- **Dosya:** `multitracker.py:31`
- **Sorun:** `np.asarray(tlwh, dtype=np.float)` — `np.float` NumPy 1.24'te kaldırıldı.
- **Etki:** Modern NumPy ile `AttributeError`. `np.float64` olmalı.

#### BUG-04: `TripletLoss` Tek Kimlikli Batch'te Çöker
- **Dosya:** `losses.py:161`
- **Sorun:** Tüm örnekler aynı kimliğe sahipse `dist[i][mask[i] == 0]` boş tensor döner; `.min()` çağrısı `RuntimeError` fırlatır.
- **Etki:** Eğitim sırasında küçük batch'lerde veya sınırlı kimlik sayısıyla rastgele crash.

#### BUG-05: `EmbeddingLoss` — `math.log(0)` ve Sıfıra Bölme
- **Dosya:** `losses.py:176, 206`
- **Sorun:** `nID=1` olduğunda `math.log(0)` = `-inf`; `id_output.size(0)=0` olduğunda sıfıra bölme.
- **Etki:** Belirli dataset konfigürasyonlarında eğitim çöker.

#### BUG-06: `torch.cuda.synchronize()` CPU'da Çöker
- **Dosya:** `detector.py:140, 339, 345, 349`
- **Sorun:** CUDA cihazı yoksa `RuntimeError` fırlatır.
- **Etki:** CPU-only ortamlarda inference yapılamaz.

#### BUG-07: Mutable Default Argument (Sessiz Veri Bozulması)
- **Dosya:** `detector.py:56, 208`
- **Sorun:** `def run(self, ..., meta={})` — Python'da değiştirilebilir varsayılan argüman paylaşılır.
- **Etki:** Ardışık çağrılarda `meta` dict'i önceki çağrılardan kalma verileri taşır.

---

### 🟠 SEVİYE: YÜKSEK

#### BUG-08: `cv2.imread` None Kontrolü Yok
- **Dosya:** `detector.py:67`, `generic_dataset.py:170`
- **Sorun:** Dosya yoksa veya bozuksa `cv2.imread` `None` döner → sonraki `.shape` erişimi `AttributeError`.

#### BUG-09: `torch.log(0)` — NaN/Inf Gradyanlar
- **Dosya:** `losses.py:20, 44`
- **Sorun:** Sigmoid çıkışı tam 0 veya 1 olduğunda `torch.log(0) = -inf`. Clamping yok.
- **Etki:** Eğitim sırasında gradyan patlaması; NaN loss.

#### BUG-10: `torch.set_grad_enabled(False)` Exception-Safe Değil
- **Dosya:** `trainer.py:164-171`
- **Sorun:** Validasyon sırasında exception fırlarsa `set_grad_enabled(True)` hiç çağrılmaz. Sonraki tüm eğitim gradyansız kalır.
- **Düzeltme:** `with torch.no_grad():` context manager kullanılmalı.

#### BUG-11: Kalman Filter Cholesky Çökmesi
- **Dosya:** `kalman_filter.py:215, 262`
- **Sorun:** Kovaryans matrisi pozitif-tanımlı olmazsa `LinAlgError` fırlatır. try/except yok.
- **Tetikleyici:** Yüksekliği sıfır/negatif olan track'ler (gürültü terimleri sıfır olur).

#### BUG-12: `update_features` Sıfır Vektör Bölme
- **Dosya:** `multitracker.py:45`
- **Sorun:** `feat /= np.linalg.norm(feat)` — All-zero feature vektörüyle `NaN` üretir. EMA ile yayılır.

#### BUG-13: Optimizer State Hiç Yüklenmiyor
- **Dosya:** `model.py:76-77`
- **Sorun:** `optimizer.load_state_dict()` satırı comment-out edilmiş. Resume edilen eğitimde momentum/Adam istatistikleri sıfırdan başlar.
- **Etki:** Eğitim sürekliliği bozulur, convergence yavaşlar.

#### BUG-14: `predict()` Aktive Edilmemiş Track'te Çöker
- **Dosya:** `multitracker.py:55`
- **Sorun:** `self.mean = None` iken `self.mean.copy()` çağrılırsa `AttributeError`.

#### BUG-15: `draw_msra_gaussian` — Width/Height Ters
- **Dosya:** `image.py:190`
- **Sorun:** `w, h = heatmap.shape[0], heatmap.shape[1]` — `shape[0]` aslında height, `shape[1]` width. Kare olmayan heatmap'lerde yanlış sınır kontrolü.

#### BUG-16: `generic_post_process` Tutarsız Return Tipi
- **Dosya:** `post_process.py:47 vs 117`
- **Sorun:** Boş durumda `([{}], [{}])` tuple, normal durumda tek list döner. Downstream'de `KeyError` riski.

#### BUG-17: Freeze-Load Sırası Yanlış
- **Dosya:** `main.py:98, 104`
- **Sorun:** Model bileşenleri önce freeze ediliyor, sonra checkpoint yükleniyor. Yükleme freeze'i geçersiz kılar.

#### BUG-18: `test.py` — `load_results` Tanımsız Değişken
- **Dosya:** `test.py:163`
- **Sorun:** `opt.load_results` boşken tracking aktifse ve `frame_id == 1` ise `NameError`.

#### BUG-19: `_to_list` Eksik NumPy Tip Desteği
- **Dosya:** `test.py:188`
- **Sorun:** Sadece `np.ndarray` ve `np.float32` dönüştürülüyor. `np.int64`, `np.float64` gibi tipler `json.dump`'da `TypeError` fırlatır.

---

### 🟡 SEVİYE: ORTA

#### BUG-20: `removed_stracks` Sınırsız Büyüme (Bellek Sızıntısı)
- **Dosya:** `multitracker.py:366`
- **Sorun:** Silinen track'ler hiç temizlenmiyor. Uzun videolarda bellek doğrusal olarak artar ve `sub_stracks` her frame'de tüm listeyi tarar.

#### BUG-21: `decode.py` Margin Hesaplaması Bağımlı Değişken Kullanıyor
- **Dosya:** `decode.py:64-67`
- **Sorun:** `l` güncellendikten sonra `r` hesaplamasında güncellenmiş `l` kullanılıyor. Asimetrik margin.

#### BUG-22: `compute_bin_loss` Mask'ı Input'a Uyguluyor, Loss'a Değil
- **Dosya:** `losses.py:91-92`
- **Sorun:** `output = output * mask.float()` yapılıyor ama `F.cross_entropy` maskelenmiş pozisyonlar için de loss hesaplıyor.

#### BUG-23: `embedding_filter` Geçmişi Bozuyor
- **Dosya:** `matching.py:212`
- **Sorun:** `embedding_history[tid][-1] = smoothed_embedding` — Raw embedding yerine smoothed değer kaydedilince bileşik smoothing etkisi oluşuyor.

#### BUG-24: `BaseTrack._count` Sıfırlanmıyor
- **Dosya:** `basetrack.py:13`
- **Sorun:** Birden fazla video işlenirken track ID'leri önceki videodan devam eder. Evaluation tool'ları bozulabilir.

#### BUG-25: `cls_id` Filtre Eşiği Tutarsızlığı
- **Dosya:** `generic_dataset.py:138` (`-999`) vs `generic_dataset.py:217` (`-99`)
- **Sorun:** Aynı amaçlı filtreleme farklı eşiklerle yapılıyor.

#### BUG-26: Scale Argümanı pre_process'e İletilmiyor
- **Dosya:** `detector.py:213-214`
- **Sorun:** `_transform_scale(image)` `scale` parametresini almıyor. Multi-scale testing fiilen çalışmaz.

#### BUG-27: Multi-Scale Açıkça Desteklenmiyor
- **Dosya:** `detector.py:373`
- **Sorun:** `merge_outputs` tek scale assert ediyor. `test_scales` döngüsü ölü kod.

---

## 3. COVER EDİLMEMİŞ EDGE CASE'LER

### 3.1 Tracker Edge Case'leri

| # | Edge Case | Durum | Açıklama |
|---|-----------|-------|----------|
| EC-01 | İlk frame'de sıfır detection | ⚠️ Kısmen | Track'ler `is_activated=False` kalır, 2. frame'de kaybolabilir |
| EC-02 | Tüm track'lerin kaybolması | ⚠️ Eksik | `removed_stracks` sınırsız büyür |
| EC-03 | `buffer_size = 0` (düşük FPS) | ❌ Yok | `frame_rate=1, track_buffer=10` → `buffer_size=0` → anında silme |
| EC-04 | Sıfır yükseklikli bounding box | ❌ Yok | `tlwh_to_xyah` sıfıra bölme + Kalman çökmesi |
| EC-05 | All-zero embedding vektörü | ❌ Yok | Normalize'da NaN, EMA ile yayılır |
| EC-06 | Negatif Kalman state (yükseklik) | ❌ Yok | Gürültü terimleri negatif/sıfır → kovaryans singular |
| EC-07 | Çok uzun video (>100K frame) | ❌ Yok | `removed_stracks` bellek sızıntısı |
| EC-08 | Thread-safe olmayan ID üretimi | ⚠️ | `BaseTrack._count += 1` atomik değil |

### 3.2 Model/Training Edge Case'leri

| # | Edge Case | Durum | Açıklama |
|---|-----------|-------|----------|
| EC-09 | Tek kimlikli batch | ❌ Yok | TripletLoss boş tensörde `.min()` çöker |
| EC-10 | nID = 0 veya 1 | ❌ Yok | `math.log(0)` veya `math.log(-1)` |
| EC-11 | Boş pozitif maske | ❌ Yok | `id_output.size(0) = 0` → sıfıra bölme |
| EC-12 | Sigmoid çıkışı tam 0 veya 1 | ❌ Yok | `torch.log(0) = -inf` |
| EC-13 | Checkpoint'te 'epoch' key yok | ❌ Yok | `KeyError` crash |
| EC-14 | Bilinmeyen architecture string | ❌ Yok | `_network_factory` KeyError |
| EC-15 | `opt = None` ile decode çağrısı | ❌ Yok | `opt.zero_tracking` AttributeError |

### 3.3 Dataset/Inference Edge Case'leri

| # | Edge Case | Durum | Açıklama |
|---|-----------|-------|----------|
| EC-16 | Bozuk/eksik görüntü dosyası | ❌ Yok | `cv2.imread` None dönerse crash |
| EC-17 | Boş `img_ids` listesi | ❌ Yok | `np.random.choice(0)` ValueError |
| EC-18 | `embedding` key'i olmayan annotation | ❌ Yok | `know_dist_weight > 0` iken KeyError |
| EC-19 | Track ID > 2²⁴ | ❌ Yok | float32 cast'te precision kaybı |
| EC-20 | Negatif detection center | ❌ Yok | Gaussian çizimde negatif indeks |
| EC-21 | Çok küçük görüntü (border > size) | ❌ Yok | `np.random.randint(low >= high)` crash |
| EC-22 | CPU-only ortam | ❌ Yok | `torch.cuda.synchronize()` crash |

---

## 4. TASARIM SORUNLARI

### 4.1 Kod Duplikasyonu
- `GenericNetwork` ve `BaseModel` head oluşturma kodu tamamen tekrarlanmış.

### 4.2 Ölü Kod
- `EmbeddingVectorLoss` (MSE-tabanlı) `trainer.py:30`'da yaratılıyor ama hiç kullanılmıyor. Sadece `cosineSimloss` aktif.
- Multi-scale test döngüsü (`detector.py`) fiilen devre dışı.

### 4.3 Import Yolu Tutarsızlığı
- `multitracker.py:21`: `from tracker import matching` — ama dosya `tracker_fair/` altında. Yanlış modülün import edilme riski.

### 4.4 Wildcard Import'lar
- `multitracker.py:11, 17`: `from models import *`, `from tracking_utils.utils import *` — isim çakışması riski.

---

## 5. ÖZET TABLO

| Kategori | Kritik | Yüksek | Orta | Toplam |
|----------|--------|--------|------|--------|
| Runtime Hataları | 7 | 12 | 8 | 27 |
| Edge Case Eksikleri | - | - | - | 22 |
| Tasarım Sorunları | - | - | - | 4 |
| **TOPLAM** | **7** | **12** | **8** | **53** |

---

## 6. ÖNCELİKLİ DÜZELTME ÖNERİLERİ

1. **BUG-01** (Kritik): `embedding_distance` çağrı sırasını düzeltin → tracking doğruluğunu doğrudan etkiler
2. **BUG-02** (Kritik): `BaseModel.__init__`'e `self.opt = opt` ekleyin
3. **BUG-03** (Kritik): `np.float` → `np.float64`
4. **BUG-09** (Yüksek): Loss fonksiyonlarına `torch.clamp(pred, 1e-7, 1-1e-7)` ekleyin
5. **BUG-06** (Kritik): CUDA çağrılarını `if torch.cuda.is_available()` ile guard'layın
6. **BUG-11** (Yüksek): Kalman filter'a try/except + kovaryans reset mekanizması ekleyin
7. **BUG-20** (Orta): `removed_stracks`'i belirli bir frame sayısından sonra temizleyin
8. **BUG-10** (Yüksek): `torch.set_grad_enabled(False)` → `with torch.no_grad():`
