"""
Video Etiketleme - COCO + YOLO-World (Birleşik) - Video Bazlı Klasörleme + Tekrar Filtreleme
-------------------------------------------------------------------------------------------------
Videodan belirli aralıklarla kare çıkarır, COCO modeli (80 sabit sınıf) ve
isteğe bağlı YOLO-World (elle veya dosyadan verilen ek kelimeler) ile
nesneleri tespit eder.

ÖNEMLİ - VİDEO BAZLI KLASÖRLEME:
Her video, --output ile verilen ANA klasörün altında KENDİ DOSYA ADIYLA
otomatik bir alt klasöre yazılır. Böylece farklı videoları aynı --output
ile çalıştırsanız bile çıktılar birbirine KARIŞMAZ.

Örnek:
    python auto_label_video.py --video kafe.mp4   --output cikti
    -> cikti/kafe/...

    python auto_label_video.py --video sokak.mp4  --output cikti
    -> cikti/sokak/...

Eğer aynı video için tekrar çalıştırırsanız ve klasör zaten doluysa,
üzerine yazmamak için otomatik olarak "_2", "_3" gibi bir sayı eklenir
(isterseniz --overwrite ile üzerine yazmayı zorlayabilirsiniz).

YENİ - SINIF BAZLI TEKRAR (DUPLICATE) FİLTRELEME:
Sabit duran bir nesne (örn. masadaki bir kamera) art arda birçok karede
neredeyse birebir aynı görünür. Bunu önlemek için, bir sınıfa ait yeni bir
kare kaydedilmeden önce, O SINIFIN en son kaydedilen görüntüsüyle
karşılaştırılır (histogram tabanlı benzerlik). Eğer yeterince benziyorsa
(varsayılan eşik: 0.97) yeni kayıt YAPILMAZ, atlanır. Böylece "kafe.mp4"
örneğinde olduğu gibi, sabit duran bir defterin/kameranın 5 kez değil,
sadece sahne gerçekten değiştiğinde yeniden kaydedilmesi sağlanır. Bu
kontrol sınıf bazlıdır -- "camera" kendi geçmişiyle, "person" kendi
geçmişiyle karşılaştırılır, birbirini etkilemez. --dedup-threshold ile
ayarlanabilir (1'e yaklaştıkça daha katı/az filtreleme yapar).

DAVRANIŞ:
- Görüntü KIRPILMAZ. Tespit edilen her nesnenin etrafına kutu çizilir.
- Bir karede AYNI sınıftan birden fazla nesne varsa -> hepsi TEK kopyada
  birlikte kutulanır, o sınıfın klasörüne kaydedilir.
- Bir karede FARKLI sınıflar varsa -> her sınıf için AYRI bir kopya üretilir
  (her kopyada sadece o sınıfın kutuları çizili), kendi sınıf klasörüne gider.
- Bir karede HİÇBİR nesne tespit edilmezse -> kare "unlabeled/" klasörüne
  (kutu çizilmeden, ham haliyle) kaydedilir.
- Bir sınıfın yeni karesi, o sınıfın son kaydedilen karesine ÇOK benziyorsa
  -> KAYDEDİLMEZ (tekrar filtrelemesi).

Çıktı yapısı (video adı "kafe.mp4" ise, --output "cikti" ile):
    cikti/
    └── kafe/
        ├── frames/                 (ham, işlenmemiş TÜM örneklenen kareler)
        ├── labeled_frames/
        │   ├── person/              (person var -> sadece person kutuları çizili, tekrar filtrelenmiş)
        │   ├── camera/
        │   └── ...
        ├── unlabeled/               (hiçbir sınıf tespit edilemeyen kareler)
        ├── labels/                  (YOLO formatında .txt etiketler, tüm sınıflar birlikte)
        ├── classes.txt              (birleşik sınıf listesi, id -> isim)
        └── detections.csv           (özet rapor: kare, sınıf, güven skoru, kaynak model)

Kurulum:
    pip install ultralytics opencv-python

Kullanım (sadece COCO):
    python auto_label_video.py --video kafe.mp4 --output cikti

Kullanım (COCO + YOLO-World, elle yazılan ek sınıflarla):
    python auto_label_video.py --video kafe.mp4 --output cikti \\
        --use-world --world-classes "hardhat,drill,red backpack"

Kullanım (COCO + YOLO-World, bir dosyadan okunan ek sınıflarla):
    python auto_label_video.py --video kafe.mp4 --output cikti \\
        --use-world --world-classes-file vocab.txt

Kullanım (tekrar filtreleme eşiğini değiştirerek):
    python auto_label_video.py --video kafe.mp4 --output cikti --dedup-threshold 0.95
"""

import argparse
import csv
import os

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Video etiketleme (COCO + YOLO-World)")
    parser.add_argument("--video", type=str, required=True, help="Girdi video dosyasının yolu")
    parser.add_argument("--output", type=str, default="output",
                         help="Ana çıktı klasörü (her video kendi adıyla bunun altına alt klasör açar)")
    parser.add_argument("--overwrite", action="store_true",
                         help="Aynı isimde video klasörü zaten varsa üzerine yaz "
                              "(verilmezse otomatik olarak _2, _3 gibi bir ek eklenir)")

    parser.add_argument("--coco-model", type=str, default="yolov8n.pt",
                         help="COCO tabanlı model (varsayılan: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.4, help="COCO için minimum güven skoru")

    parser.add_argument("--use-world", action="store_true",
                         help="YOLO-World modelini de devreye al (ek/özel sınıflar için)")
    parser.add_argument("--world-model", type=str, default="yolov8s-world.pt",
                         help="YOLO-World model dosyası")
    parser.add_argument("--world-classes", type=str, default="",
                         help="YOLO-World sınıfları, virgülle ayrılmış, elle yazarak")
    parser.add_argument("--world-classes-file", type=str, default="",
                         help="Ek sınıfları içeren bir dosyanın yolu (elle yazmak yerine)")
    parser.add_argument("--world-conf", type=float, default=0.15,
                         help="YOLO-World için güven eşiği")

    parser.add_argument("--frame-interval-sec", type=float, default=1.0,
                         help="Kaç saniyede bir kare alınacak")
    parser.add_argument("--iou-dedup", type=float, default=0.6,
                         help="İki modelin aynı nesneyi iki kez işaretlemesini önlemek için IoU eşiği")

    parser.add_argument("--dedup-threshold", type=float, default=0.97,
                         help="Sınıf bazlı tekrar filtreleme benzerlik eşiği (0-1 arası). "
                              "Bir sınıfın yeni karesi, o sınıfın son kaydedilen karesine bu "
                              "orandan fazla benziyorsa kaydedilmez. 1'e yaklaştıkça daha az "
                              "filtreleme yapılır (sadece neredeyse birebir aynı kareler elenir).")
    parser.add_argument("--disable-dedup", action="store_true",
                         help="Sınıf bazlı tekrar filtrelemeyi tamamen kapat "
                              "(her tespit ayrı ayrı kaydedilir, eski davranış)")
    return parser.parse_args()


def get_video_output_dir(output_root, video_path, overwrite):
    """
    Video dosyasının adına göre (uzantısız) benzersiz bir alt klasör yolu üretir.
    Böylece farklı videoların çıktıları birbirine karışmaz.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_name = video_name.strip().replace(" ", "_")

    candidate = os.path.join(output_root, video_name)

    if overwrite or not os.path.exists(candidate):
        return candidate

    # Klasör zaten varsa ve üzerine yazma istenmiyorsa, _2, _3... ekle
    counter = 2
    while os.path.exists(f"{candidate}_{counter}"):
        counter += 1
    return f"{candidate}_{counter}"


def setup_output_dirs(video_output_dir):
    frames_dir = os.path.join(video_output_dir, "frames")
    labeled_dir = os.path.join(video_output_dir, "labeled_frames")
    unlabeled_dir = os.path.join(video_output_dir, "unlabeled")
    labels_dir = os.path.join(video_output_dir, "labels")

    for d in [frames_dir, labeled_dir, unlabeled_dir, labels_dir]:
        os.makedirs(d, exist_ok=True)

    return frames_dir, labeled_dir, unlabeled_dir, labels_dir


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(area1 + area2 - inter_area)


def image_similarity(img_a, img_b):
    """
    İki görüntü arasındaki benzerliği histogram karşılaştırmasıyla hesaplar.
    1.0 -> birebir aynı, 0.0 -> tamamen farklı. Basit ve hızlı bir yöntem,
    amaç kusursuz ölçüm değil, "neredeyse aynı kareyi" ucuza ayıklamak.
    """
    hist_a = cv2.calcHist([img_a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_b = cv2.calcHist([img_b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)


class UnifiedClassRegistry:
    def __init__(self):
        self.name_to_id = {}
        self.id_to_name = {}

    def get_id(self, name):
        name = name.strip().lower().replace(" ", "_")
        if name not in self.name_to_id:
            new_id = len(self.name_to_id)
            self.name_to_id[name] = new_id
            self.id_to_name[new_id] = name
        return self.name_to_id[name]

    def save(self, path):
        with open(path, "w") as f:
            for i in range(len(self.id_to_name)):
                f.write(f"{i} {self.id_to_name[i]}\n")


def run_detector(model, frame, conf, source_tag):
    detections = []
    results = model.predict(frame, conf=conf, verbose=False)[0]
    names = model.names
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        conf_score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "name": cls_name,
            "conf": conf_score,
            "box": (x1, y1, x2, y2),
            "source": source_tag,
        })
    return detections


def merge_detections(coco_dets, world_dets, iou_threshold):
    all_dets = coco_dets + world_dets
    keep = []
    used = [False] * len(all_dets)
    order = sorted(range(len(all_dets)), key=lambda i: all_dets[i]["conf"], reverse=True)

    for i in order:
        if used[i]:
            continue
        keep.append(all_dets[i])
        for j in order:
            if used[j] or i == j:
                continue
            same_name = all_dets[i]["name"].lower() == all_dets[j]["name"].lower()
            iou = box_iou(all_dets[i]["box"], all_dets[j]["box"])
            if same_name and iou >= iou_threshold:
                used[j] = True

    return keep


def draw_boxes(frame, dets):
    """Verilen tespit listesinin kutularını bir kopya üzerine çizer."""
    annotated = frame.copy()
    for det in dets:
        x1, y1, x2, y2 = det["box"]
        color = (0, 255, 0) if det["source"] == "coco" else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det['name']} {det['conf']:.2f}"
        cv2.putText(annotated, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video bulunamadı: {args.video}")

    video_output_dir = get_video_output_dir(args.output, args.video, args.overwrite)
    print(f"Bu videonun çıktıları şuraya yazılacak: {os.path.abspath(video_output_dir)}")

    frames_dir, labeled_dir, unlabeled_dir, labels_dir = setup_output_dirs(video_output_dir)
    registry = UnifiedClassRegistry()

    print(f"COCO modeli yükleniyor: {args.coco_model}")
    coco_model = YOLO(args.coco_model)

    world_model = None
    if args.use_world:
        world_class_list = []
        if args.world_classes_file:
            with open(args.world_classes_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            world_class_list = [c.strip() for c in content.split(",") if c.strip()]
        elif args.world_classes:
            world_class_list = [c.strip() for c in args.world_classes.split(",") if c.strip()]

        if not world_class_list:
            raise ValueError("--use-world verildi ama --world-classes veya "
                              "--world-classes-file ile hiç sınıf sağlanmadı.")

        print(f"YOLO-World modeli yükleniyor: {args.world_model}")
        print(f"Aranacak ek sınıf sayısı: {len(world_class_list)}")
        world_model = YOLO(args.world_model)
        world_model.set_classes(world_class_list)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25
    frame_interval = max(1, int(fps * args.frame_interval_sec))

    print(f"Video FPS: {fps:.2f}")
    print(f"Her {frame_interval} karede bir işlem yapılacak "
          f"(~{args.frame_interval_sec} saniyede 1 kare)")
    if not args.disable_dedup:
        print(f"Sınıf bazlı tekrar filtreleme AÇIK (eşik: {args.dedup_threshold})")
    else:
        print("Sınıf bazlı tekrar filtreleme KAPALI")

    frame_idx = 0
    saved_idx = 0
    class_counters = {}
    csv_rows = []  # (frame_id, class, confidence, source)

    # Her sınıfın en son KAYDEDİLEN annotated görüntüsünü tutar -> tekrar
    # filtrelemesi için karşılaştırma referansı
    last_saved_by_class = {}
    skipped_duplicates = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            h, w = frame.shape[:2]
            frame_filename = f"frame_{saved_idx:06d}.jpg"
            cv2.imwrite(os.path.join(frames_dir, frame_filename), frame)

            coco_dets = run_detector(coco_model, frame, args.conf, "coco")
            world_dets = []
            if world_model is not None:
                world_dets = run_detector(world_model, frame, args.world_conf, "world")

            merged = merge_detections(coco_dets, world_dets, args.iou_dedup)

            if not merged:
                # --- Hiçbir nesne tespit edilmedi -> unlabeled klasörüne ---
                cv2.imwrite(os.path.join(unlabeled_dir, frame_filename), frame)
            else:
                # --- YOLO formatında etiket dosyası (tüm sınıflar birlikte) ---
                yolo_lines = []
                for det in merged:
                    x1, y1, x2, y2 = det["box"]
                    cls_id = registry.get_id(det["name"])
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    box_w = (x2 - x1) / w
                    box_h = (y2 - y1) / h
                    yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

                    csv_rows.append((frame_filename, det["name"], f"{det['conf']:.3f}", det["source"]))

                label_filename = frame_filename.replace(".jpg", ".txt")
                with open(os.path.join(labels_dir, label_filename), "w") as f:
                    f.write("\n".join(yolo_lines))

                # --- Sınıf bazlı ayrı kopyalar (tekrar filtrelemeli) ---
                dets_by_class = {}
                for det in merged:
                    dets_by_class.setdefault(det["name"], []).append(det)

                for cls_name, class_dets in dets_by_class.items():
                    safe_name = cls_name.strip().lower().replace(" ", "_")
                    annotated = draw_boxes(frame, class_dets)

                    # --- Tekrar (duplicate) kontrolü: bu sınıfın son kaydedilen
                    #     karesine çok mu benziyor? ---
                    if not args.disable_dedup and safe_name in last_saved_by_class:
                        sim = image_similarity(last_saved_by_class[safe_name], annotated)
                        if sim >= args.dedup_threshold:
                            skipped_duplicates.setdefault(safe_name, 0)
                            skipped_duplicates[safe_name] += 1
                            continue  # çok benziyor, kaydetme, atla

                    class_dir = os.path.join(labeled_dir, safe_name)
                    os.makedirs(class_dir, exist_ok=True)

                    class_counters.setdefault(safe_name, 0)
                    class_counters[safe_name] += 1

                    avg_conf = sum(d["conf"] for d in class_dets) / len(class_dets)
                    conf_tag = int(avg_conf * 100)
                    out_filename = f"{safe_name}_{class_counters[safe_name]:05d}_conf{conf_tag}.jpg"
                    cv2.imwrite(os.path.join(class_dir, out_filename), annotated)

                    # Bu sınıfın "son kaydedilen" referansını güncelle
                    last_saved_by_class[safe_name] = annotated

            saved_idx += 1
            if saved_idx % 20 == 0:
                print(f"{saved_idx} kare işlendi...")

        frame_idx += 1

    cap.release()

    registry.save(os.path.join(video_output_dir, "classes.txt"))

    csv_path = os.path.join(video_output_dir, "detections.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "class", "confidence", "source"])
        writer.writerows(csv_rows)

    print("\nTamamlandı.")
    print(f"Toplam işlenen kare: {saved_idx}")
    print(f"Sınıf bazlı KAYDEDİLEN görüntü sayıları: {class_counters}")
    if skipped_duplicates:
        print(f"Sınıf bazlı ATLANAN (tekrar/duplicate) görüntü sayıları: {skipped_duplicates}")
    print(f"Çıktılar: {os.path.abspath(video_output_dir)}")
    print(f"Özet rapor: {csv_path}")
    print(f"Sınıf listesi: {os.path.join(video_output_dir, 'classes.txt')}")


if __name__ == "__main__":
    main()