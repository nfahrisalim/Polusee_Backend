import cv2
from collections import defaultdict, Counter
from ultralytics import YOLO

def vehicle_counter():
    print("Load model custom...")
    model = YOLO('best.pt')

    cap = cv2.VideoCapture(0) # camera iphone

    if not cap.isOpened():
        print("Can't open camera...")
        return

    # Counter
    # total_kendaraan = {2: 0, 3: 0, 5: 0, 7: 0}
    # nama_kendaraan = {2: "Mobil", 3: "Motor", 5: "Bus", 7: "Truk"}

    # Counter
    total_kendaraan = {0: 0, 1: 0, 2: 0, 3: 0}
    nama_kendaraan = {0: "Bus", 1: "Mobil", 2: "Motor", 3: "Truk"}

    id_sudah_dihitung = set()
    histori_kelas = defaultdict(list)

    # syarat jumlah frame diturunkan agar motor cepat tetap terekam
    MINIMAL_FRAME_VOTING = 4

    print("'q' for exit...'")

    while True:
        berhasil, frame = cap.read()
        if not berhasil:
            print("Terminated...")
            break

        tinggi, lebar, _ = frame.shape

        # 1. Area fokus (RoI)
        y1, y2 = int(tinggi * 0.1), int(tinggi * 0.9)
        x1, x2 = int(lebar * 0.1), int(lebar * 0.9)
        area_fokus = frame[y1:y2, x1:x2]

        # tracking pake YOLO
        hasil = model.track(
            source=area_fokus,
            # classes=[2, 3, 5, 7],
            classes=[0, 1, 2, 3],
            conf=0.40,
            persist=True,
            tracker="bytetrack.yaml", # masi pelajari better pakai ini atau BoT-SORT
            verbose=False
        )

        frame_hasil_yolo = hasil[0].plot()
        frame[y1:y2, x1:x2] = frame_hasil_yolo

        boxes = hasil[0].boxes

        # simpen ID yg ada saat ini
        id_saat_ini = set()

        # hitung kendaraan
        if boxes.id is not None:
            id_tracking = boxes.id.cpu().numpy().astype(int)
            kelas_objek = boxes.cls.cpu().numpy().astype(int)

            # Masukkan semua ID yang ada di layar ke set
            id_saat_ini = set(id_tracking)

            for i in range(len(id_tracking)):
                id_obj = id_tracking[i]
                id_kelas = kelas_objek[i]

                histori_kelas[id_obj].append(id_kelas)

                # kendaraan lemot n normal
                if len(histori_kelas[id_obj]) >= MINIMAL_FRAME_VOTING:
                    if id_obj not in id_sudah_dihitung:

                        frame_terakhir = histori_kelas[id_obj][-5:] # ambil 5 frame terakhir
                        kelas_akhir = Counter(frame_terakhir).most_common(1)[0][0]

                        total_kendaraan[kelas_akhir] += 1
                        id_sudah_dihitung.add(id_obj)
                        print(f"[ID: {id_obj}] normal: Counted as {nama_kendaraan[kelas_akhir]}")

        # buat kendaraan yg cpt
        for id_obj in list(histori_kelas.keys()):
            if id_obj not in id_saat_ini and id_obj not in id_sudah_dihitung:

                # ambil kelas terakhir dari histori
                kelas_akhir = histori_kelas[id_obj][-1]

                total_kendaraan[kelas_akhir] += 1
                id_sudah_dihitung.add(id_obj)
                print(f"[ID: {id_obj}] ngebut: Counted as {nama_kendaraan[kelas_akhir]}")

        # Visualization

        # RoI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "Detection area", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Scoreboard
        cv2.rectangle(frame, (10, 10), (250, 160), (0, 0, 0), -1)
        cv2.putText(frame, "Total kendaraan", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y_text = 65
        for id_kelas, total in total_kendaraan.items():
            teks = f"{nama_kendaraan[id_kelas]}: {total}"
            cv2.putText(frame, teks, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_text += 30

        cv2.imshow("Vehicle counter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    vehicle_counter()
