import cv2
import mediapipe as mp
import time
import handtrackingmodule as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        # Print all landmarks for the detected hand
        print("Landmarks for all fingers:")
        for landmark in lmList:
            print(f"ID: {landmark[0]}, X: {landmark[1]}, Y: {landmark[2]}")

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()