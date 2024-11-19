from ultralytics import YOLO
from ultralytics import solutions
import cv2


# model = YOLO('boxes.pt')

# result = model(source=r"C:\Users\Awais Shakeel\Downloads\1092314411-preview.mp4", conf=0.25, save=True, show=True, show_conf=False)



cap = cv2.VideoCapture(r"C:\Users\Awais Shakeel\Downloads\1076570246-preview.mp4")

assert cap.isOpened()


w,h,fps =  (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

region_point = [(20,150),(20,300)]
# region_point = [(20, 100), (300, 100), (300, 20), (20, 20)]
# region_point = [(20, 600), (300, 600), (300, 20), (20, 20)]

video_writer = cv2.VideoWriter('box_count.avi', cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

counter = solutions.ObjectCounter(
    model = 'boxes.pt',
    region=region_point,
    show=True,
    show_in=True,
    show_out= False, 
    conf=0.35
)



while cap.isOpened():
    success , frame = cap.read()
    if success:
        frame  = counter.count(frame)
        video_writer.write(frame)
        
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    else:
        break    
        
    
cap.release()    
video_writer.release()
cv2.destroyAllWindows()
