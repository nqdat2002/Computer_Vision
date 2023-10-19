import cv2 as cv
def TestCam():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

def TestPlayVideo():
    cap = cv.VideoCapture('file_example_AVI_480_750kB.avi')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

def TestSaveVideo():
    cap = cv.VideoCapture(0)
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    out = cv.VideoWriter('out.avi', fourcc, 30.0, (640, 480))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv.flip(frame, 0)
        out.write(frame)
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    out.release()
    cv.destroyAllWindows()

def main():
    options = int(input())
    while 1:
        if options == 1:
            TestCam()
            break
        if options == 2:
            TestPlayVideo()
            break
        if options == 3:
            TestSaveVideo()
            break

if __name__ == '__main__':
    main()