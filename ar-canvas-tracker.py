import cv2
import numpy as np
from collections import deque


class ArucoDocumentTracker:
    def __init__(self, buffer_size=5):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.adaptiveThreshConstant = 7
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.minMarkerPerimeterRate = 0.03
        self.aruco_params.maxMarkerPerimeterRate = 0.5
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.marker_size = 0.025
        self.corners_buffer = deque(maxlen=buffer_size)
        self.prev_corners = None


    def generate_markers(self, output_dir="markers"):
        """四隅に配置するARマーカーを生成する"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        marker_size = 400  # マーカーサイズを大きく
        marker_ids = [0, 1, 2, 3]

        for marker_id in marker_ids:
            marker_image = self.aruco_dict.generateImageMarker(marker_id, marker_size, borderBits=2)
            filename = f"{output_dir}/marker_{marker_id}.png"
            cv2.imwrite(filename, marker_image)

        # A4テンプレートの生成
        a4_template = np.ones((2970, 2100), dtype=np.uint8) * 255  # A4 size at 300dpi

        # マーカーの配置位置を内側に移動
        margin = 100  # マージンを設定
        marker_positions = [
            (margin, margin),  # 左上
            (2100-marker_size-margin, margin),  # 右上
            (2100-marker_size-margin, 2970-marker_size-margin),  # 右下
            (margin, 2970-marker_size-margin)  # 左下
        ]

        for i, pos in enumerate(marker_positions):
            marker_image = self.aruco_dict.generateImageMarker(i, marker_size, borderBits=2)
            x, y = pos
            a4_template[y:y+marker_size, x:x+marker_size] = marker_image

        cv2.imwrite(f"{output_dir}/a4_template.png", a4_template)
        print(f"マーカーを生成しました: {output_dir}/a4_template.png")


    def detect_markers(self, frame):
        """フレーム内のARマーカーを検出する"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # コントラストを強調
        gray = cv2.equalizeHist(gray)

        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

        # 検出されたマーカーの情報を辞書形式で返す
        detected_markers = {}
        if ids is not None:
            for marker_id, marker_corners in zip(ids, corners):
                detected_markers[marker_id[0]] = marker_corners[0]

        return detected_markers


    def estimate_missing_corner(self, markers):
        """3つのマーカーから欠けている1つのマーカーの位置を推定する"""
        # 存在するマーカーのIDと位置を取得
        existing_ids = list(markers.keys())
        missing_id = list(set([0, 1, 2, 3]) - set(existing_ids))[0]

        # マーカーの配置は以下の通り：
        # 0 (左上) - 1 (右上)
        # 3 (左下) - 2 (右下)

        # 欠けているコーナーのパターンごとに推定方法を変える
        if missing_id == 0:    # 左上が欠けている
            if all(i in existing_ids for i in [1, 2, 3]):
                # 右上(1) + 左下(3) - 右下(2) = 左上(0)
                estimated_point = markers[1][0] + markers[3][0] - markers[2][0]
                return 0, estimated_point

        elif missing_id == 1:  # 右上が欠けている
            if all(i in existing_ids for i in [0, 2, 3]):
                # 左上(0) - 左下(3) + 右下(2) = 右上(1)
                estimated_point = markers[0][0] - markers[3][0] + markers[2][0]
                return 1, estimated_point

        elif missing_id == 2:  # 右下が欠けている
            if all(i in existing_ids for i in [0, 1, 3]):
                # 右上(1) - 左上(0) + 左下(3) = 右下(2)
                estimated_point = markers[1][0] - markers[0][0] + markers[3][0]
                return 2, estimated_point

        elif missing_id == 3:  # 左下が欠けている
            if all(i in existing_ids for i in [0, 1, 2]):
                # 左上(0) - 右上(1) + 右下(2) = 左下(3)
                estimated_point = markers[0][0] - markers[1][0] + markers[2][0]
                return 3, estimated_point

        return None, None


    def get_document_corners(self, detected_markers):
        """4つのマーカーから文書の角を推定する（3つの場合は推定を試みる）"""
        # 4つのマーカーが検出された場合
        if len(detected_markers) == 4:
            document_corners = np.array([
                detected_markers[0][0],  # 左上
                detected_markers[1][0],  # 右上
                detected_markers[2][0],  # 右下
                detected_markers[3][0]   # 左下
            ], dtype=np.float32)
            return document_corners, None

        # 3つのマーカーが検出された場合
        elif len(detected_markers) == 3:
            # 欠けているコーナーを推定
            missing_id, estimated_point = self.estimate_missing_corner(detected_markers)
            if missing_id is not None:
                # 推定したコーナーを含めてコーナーの配列を作成
                document_corners = []
                for i in range(4):
                    if i == missing_id:
                        document_corners.append(estimated_point)
                    else:
                        document_corners.append(detected_markers[i][0])
                return np.array(document_corners, dtype=np.float32), missing_id

        return None, None


    def draw_detections(self, frame, detected_markers, document_corners=None, estimated_id=None):
        """検出結果を描画する"""
        output = frame.copy()

        # 検出されたマーカーの数を表示
        cv2.putText(output, f"Detected markers: {len(detected_markers)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 各マーカーの情報を描画
        for marker_id, corners in detected_markers.items():
            # マーカーの輪郭を描画
            corners_int = corners.astype(int)
            for i in range(4):
                pt1 = tuple(corners_int[i])
                pt2 = tuple(corners_int[(i + 1) % 4])
                cv2.line(output, pt1, pt2, (0, 255, 0), 2)

            # マーカーのIDを描画
            center = corners_int.mean(axis=0).astype(int)
            cv2.putText(output, f"ID:{marker_id}", tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 文書の角が推定できている場合
        if document_corners is not None:
            color = (255, 255, 0)  # 通常は黄色

            # 4点を線で結ぶ
            for i in range(4):
                pt1 = tuple(document_corners[i].astype(int))
                pt2 = tuple(document_corners[(i + 1) % 4].astype(int))
                cv2.line(output, pt1, pt2, color, 2)

                # 推定された点の場合は特別な表示
                if i == estimated_id:
                    cv2.circle(output, pt1, 8, (0, 0, 255), -1)  # 赤い点
                    cv2.putText(output, "Estimated", pt1,
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return output


def main():
    tracker = ArucoDocumentTracker()

    # 印刷用マーカーを生成
    tracker.generate_markers()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラをオープンできません")
        return

    print("カメラを起動しました。'q'キーで終了します。")

    # A4サイズの出力設定
    OUTPUT_WIDTH = 595
    OUTPUT_HEIGHT = 842
    dst_points = np.array([
        [0, 0],
        [OUTPUT_WIDTH - 1, 0],
        [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
        [0, OUTPUT_HEIGHT - 1]
    ], dtype="float32")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("エラー: フレームを読み取れません")
            break

        # マーカーを検出
        detected_markers = tracker.detect_markers(frame)

        # コーナーを取得（3つのマーカーからの推定を含む）
        document_corners, estimated_id = tracker.get_document_corners(detected_markers)

        # 検出結果を描画
        output = tracker.draw_detections(frame, detected_markers, document_corners, estimated_id)

        # 4点が揃った場合（推定を含む）は変形処理を実行
        if document_corners is not None:
            matrix = cv2.getPerspectiveTransform(document_corners, dst_points)
            warped = cv2.warpPerspective(frame, matrix, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
            cv2.imshow("Warped", warped)

        cv2.imshow("Original", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("プログラムを終了します")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
