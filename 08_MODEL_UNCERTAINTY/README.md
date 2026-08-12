# UAV Model Uncertainty  - V1

This part of the project was made to check how reliable the YOLO detections really are. Normally the model gives a class and a confidence score, but this does not always mean the detection is stable. In UAV footage, a target can be far away, blurry, compressed or affected by lighting, so I wanted to see how much the result changes when the image quality changes slightly.

For this first version, I kept the current trained model exactly as it is. I did not retrain it or change the YOLO architecture. Instead, I added a separate uncertainty test around it. The system first runs the original image, then creates several slightly modified versions of the same image using small changes in brightness, contrast, blur, noise and JPEG compression.

By default, I use 10 modified images, so together with the original image there are 11 predictions in total. The goal is to check if the same target is still detected, if the predicted class stays the same, if the confidence changes a lot and if the bounding box stays in a similar position.

After all the predictions are done, the system tries to match the same target between the different images using the overlap of the bounding boxes, called IoU. I did not force the class to stay the same during matching because this is exactly one of the things I want to measure. For example, if the same target changes from `tank` to `military_vehicle` or `truck`, the system keeps it as the same target and records that the classification was unstable.

For every target, I calculate simple values that are easy to understand, such as how many times the target was detected, the average confidence, how much the confidence changes, how often the class stays the same, and how much the bounding box moves or changes size. I decided not to combine everything into one uncertainty score yet because for this first version I prefer to see the raw results and understand exactly where the model is unstable.

The whole uncertainty system is separated from the active Windows sender, ROS 2, BoT-SORT and the threat logic. This means I can test and improve it without affecting the current working pipeline. The code is inside `08_MODEL_UNCERTAINTY/`, with separate files for the image changes, the YOLO adapter, target matching, uncertainty calculations and the main runner.

The tests also work without the real YOLO model, GPU, ROS 2 or camera. This was important because it lets me test the logic separately from the full system. The random image changes can also be repeated using the same seed, so the same test can produce the same result again.

For now, the target matching uses a simple IoU-based method. It works well enough for this first version, but I already know that it can become less reliable when several targets are very close together or cross each other. This is something I can improve later if needed.

This V1 is not true Monte Carlo Dropout yet. For now, I am keeping the model unchanged and testing how stable the predictions are when the input image changes slightly. Later, I can test Monte Carlo Dropout directly inside the YOLO network and compare the two approaches.

The next step is to run this on real UAV images and compare easy detections with smaller, distant or more difficult targets. That will show whether the uncertainty values are really useful and whether they can help identify the cases that should be improved in the next training stage.
