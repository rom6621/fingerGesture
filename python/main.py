import cv2
import numpy as np
import mediapipe as mp

# MediaPipeインスタンス生成
hands = mp.solutions.hands.Hands(
  max_num_hands=1,
  min_detection_confidence=0.7,
  min_tracking_confidence=0.5,
)

# ランドマーク座標の取得を行う関数
def getLandmarks(image):
  # 映像から手を識別
  results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  if results.multi_hand_landmarks is not None:
    ret = []
    landmarks = results.multi_hand_landmarks[0].landmark
    for index, landmark in enumerate(landmarks):
      landmark_x = min(int(landmark.x*image.shape[1]), image.shape[1]-1)
      landmark_y = min(int(landmark.y*image.shape[0]), image.shape[0]-1)
      ret.append((landmark_x, landmark_y))
    return ret
  else:
    return None

# 接続線の描画を行う関数
def drawPoint(image, landmarks):
  # ランドマーク点の描画
  for landmark in landmarks:
    cv2.circle(image, landmark, 5, (255, 255, 0), 2)
  # 接続線を描画
  cv2.line(image, landmarks[2], landmarks[3], (0, 255, 0), 2)
  cv2.line(image, landmarks[3], landmarks[4], (0, 255, 0), 2)
  cv2.line(image, landmarks[5], landmarks[6], (0, 255, 0), 2)
  cv2.line(image, landmarks[6], landmarks[7], (0, 255, 0), 2)
  cv2.line(image, landmarks[7], landmarks[8], (0, 255, 0), 2)
  cv2.line(image, landmarks[9], landmarks[10], (0, 255, 0), 2)
  cv2.line(image, landmarks[10], landmarks[11], (0, 255, 0), 2)
  cv2.line(image, landmarks[11], landmarks[12], (0, 255, 0), 2)
  cv2.line(image, landmarks[13], landmarks[14], (0, 255, 0), 2)
  cv2.line(image, landmarks[14], landmarks[15], (0, 255, 0), 2)
  cv2.line(image, landmarks[15], landmarks[16], (0, 255, 0), 2)
  cv2.line(image, landmarks[17], landmarks[18], (0, 255, 0), 2)
  cv2.line(image, landmarks[18], landmarks[19], (0, 255, 0), 2)
  cv2.line(image, landmarks[19], landmarks[20], (0, 255, 0), 2)
  cv2.line(image, landmarks[0], landmarks[1], (0, 255, 0), 2)
  cv2.line(image, landmarks[1], landmarks[2], (0, 255, 0), 2)
  cv2.line(image, landmarks[2], landmarks[5], (0, 255, 0), 2)
  cv2.line(image, landmarks[5], landmarks[9], (0, 255, 0), 2)
  cv2.line(image, landmarks[9], landmarks[13], (0, 255, 0), 2)
  cv2.line(image, landmarks[13], landmarks[17], (0, 255, 0), 2)
  cv2.line(image, landmarks[17], landmarks[0], (0, 255, 0), 2)

def recognition(image, landmarks):
  # 親指先と手首の距離
  f1 = np.linalg.norm(np.array(landmarks[0])-np.array(landmarks[4]))
  # 人差し指先と手首の距離
  f2 = np.linalg.norm(np.array(landmarks[0])-np.array(landmarks[8]))
  # 中指先と手首の距離
  f3 = np.linalg.norm(np.array(landmarks[0])-np.array(landmarks[12]))
  # 薬指先と手首の距離
  f4 = np.linalg.norm(np.array(landmarks[0])-np.array(landmarks[16]))
  # 小指先と手首の距離
  f5 = np.linalg.norm(np.array(landmarks[0])-np.array(landmarks[20]))
  # 親指第1関節と手首の距離
  f6 = np.linalg.norm(np.array(landmarks[0])-np.array(landmarks[3]))
  # 人差し指第2関節と手首の距離
  f7 = np.linalg.norm(np.array(landmarks[0])-np.array(landmarks[6]))
  # 中指第2関節と手首の距離
  f8 = np.linalg.norm(np.array(landmarks[0])-np.array(landmarks[10]))
  # 薬指第2関節と手首の距離
  f9 = np.linalg.norm(np.array(landmarks[0])-np.array(landmarks[14]))
  # 小指第2関節と手首の距離
  f10 = np.linalg.norm(np.array(landmarks[0])-np.array(landmarks[18]))
  # 親指先と人差し指第2関節の距離
  f11 = np.linalg.norm(np.array(landmarks[4])-np.array(landmarks[6]))
  # 親指の付け根と手首のY座標の距離
  f12 = landmarks[3][1] - landmarks[4][1]
  # 手首と手首真ん中のY座標の距離
  f13 = landmarks[2][1] - landmarks[3][1]
  # 人差し指指先と人差し指第1関節のY座標の距離
  f14 = landmarks[1][1] - landmarks[2][1]
  # 人差し指の第1関数と人差し指の第2関数のY座標の距離
  f15 = landmarks[0][1] - landmarks[1][1]

  gestureName = ""

  if f1>f2 and f1>f3 and f1>f4 and f1>f5:
    if f12>10 and f13>0 and f14>0 and f15>0:
      gestureName = "Good"
    elif f12<0 and f13<0 and f14<0 and f15<0:
      gestureName = "Bad"
    else:
      gestureName = "Rock"
  elif f1>f6 and f2>f7 and f3>f8 and f4>f9 and f5>f10:
      gestureName = "Paper"
  elif f2>f7 and f3>f8 and f4<f9 and f5<f10:
    gestureName = "Scissors"
  elif f2<f7 and f3>f8 and f4<f9 and f5<f10:
    gestureName = "Fuxx"

  cv2.putText(image, gestureName, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

def main():
  # カメラ映像の取得
  cap = cv2.VideoCapture(0)
  while True:
    # カメラ映像の読み込み
    ret, image = cap.read()
    image = cv2.flip(image, 1)

    # ランドマーク座標の取得
    landmarks = getLandmarks(image)
    # 手が認識された場合
    if landmarks != None:
      # ポイントと線を描画
      drawPoint(image, landmarks)
      recognition(image, landmarks)

    # 描画結果を画面に表示
    cv2.imshow("Demo", image)
    # Escapeが入力された場合終了
    key = cv2.waitKey(1)
    if key == 27:
      break
  cv2.destroyAllWindows()
  cap.release()

if __name__ == "__main__":
  main()