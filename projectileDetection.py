import cv2
import numpy as np
import math

def get_color_ranges(color_name):
    color_dict = {
        'orange': (np.array([0, 150, 155]), np.array([100, 250, 255])),
        'white': (np.array([230, 230, 230]), np.array([255, 255, 255])),
        'brown': (np.array([4, 210, 47]), np.array([7, 230, 55]))
    }
    return color_dict.get(color_name.lower(), (None, None))

def screen_detect(frame, color_range, screen_part):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_color, upper_color = color_range
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    if screen_part == 'top half':
        mask = mask[0:frame.shape[0]//2, :]
    elif screen_part == 'top right':
        mask = mask[0:frame.shape[0]//2, frame.shape[1]//2:]
    elif screen_part == 'full':
        pass  # Full frame, no change to mask
    else:
        raise ValueError("Unexpected screen part input")

    return mask

def filter_contours(contours, max_distance):
    filtered_contours = []
    for i, contour in enumerate(contours):
        M1 = cv2.moments(contour)
        if M1["m00"] != 0:
            cX1, cY1 = int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"])
            for j, other_contour in enumerate(contours):
                if i != j:
                    M2 = cv2.moments(other_contour)
                    if M2["m00"] != 0:
                        cX2, cY2 = int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"])
                        distance = np.sqrt((cX2 - cX1) ** 2 + (cY2 - cY1) ** 2)
                        if distance < max_distance:
                            filtered_contours.append(contour)
                            break
    return filtered_contours

def compute_initial_values(centroidX, centroidY, framenumber, fps):
    if len(centroidX) > 1:
        dx = (centroidX[-1] - centroidX[-2]) / 395
        dy = (centroidY[-1] - centroidY[-2]) / 395
        t = (framenumber[-1] - framenumber[-2]) / fps

        vx = dx / t
        vy = dy / t

        if vx**2 + vy**2 >= 10:
            initialv = vx**2 + vy**2
            launch_angle = -np.degrees(np.arctan(dy / dx))
            initial_height = centroidY[-2] / 395

            return initialv, launch_angle, initial_height

    return 0, 0, 0

def draw_path(frame, posX, posY):
    if len(posX) > 2 and len(posY) > 2:
        posX_np = np.array(posX)
        posY_np = np.array(posY)
        coefficients = np.polyfit(posX_np, posY_np, 2)
        x_range = np.linspace(0, frame.shape[1], 1000)
        y_values = np.polyval(coefficients, x_range)

        for x, y in zip(x_range, y_values):
            cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)

def projectile_motion(initialv, launch_angle, initial_height):
    g = 9.81  # acceleration due to gravity in m/s^2
    launch_angle_rad = math.radians(launch_angle)
    v_x0 = initialv * math.cos(launch_angle_rad)
    v_y0 = initialv * math.sin(launch_angle_rad)
    max_height = initial_height + (v_y0**2) / (2 * g)
    t_f = (v_y0 + math.sqrt(v_y0**2 + 2 * g * initial_height)) / g
    range_distance = v_x0 * t_f
    v_y_final = math.sqrt(v_y0**2 + 2 * g * max_height)
    max_v = math.sqrt(v_x0**2 + v_y_final**2)
    return max_v, max_height, range_distance

def main():
    cap = cv2.VideoCapture('data/derenBasketballTest1.mp4')
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('data/output_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    color_input = input("Enter the color you want to detect: ")
    color_range = get_color_ranges(color_input)

    if color_range == (None, None):
        print("Error: Unsupported color.")
        return

    screen_part = input("Enter the part of the screen you want to detect: ")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_counter = 0
    posX, posY = [], []
    centroidX, centroidY, framenumber = [], [], []
    initialv, launch_angle, initial_height = 0, 0, 0
    max_distance = 50

    cv2.namedWindow('Resized Video Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Resized Video Window', 980, 540)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        mask = screen_detect(frame, color_range, screen_part)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = filter_contours(contours, max_distance)

        sumx, sumy, legal = 0, 0, 0
        for contour in filtered_contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                if initialv != 0:
                    posX.append(cX)
                    posY.append(cY)
                sumx += cX
                sumy += cY
                legal += 1
                if initialv != 0:
                    cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

        if legal != 0:
            centroidX.append(sumx / legal)
            centroidY.append(sumy / legal)
            framenumber.append(frame_counter)

        initialv, launch_angle, initial_height = compute_initial_values(centroidX, centroidY, framenumber, fps)
        if initialv != 0:
            draw_path(frame, posX, posY)

        out.write(frame)
        cv2.imshow('Resized Video Window', frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if initialv != 0:
        max_v, max_height, range_distance = projectile_motion(initialv, launch_angle, initial_height)
        print("Max Velocity: ", max_v, " meters per second")
        print("Max Height: ", max_height, " meters")
        print("Dist Travelled: ", range_distance, " meters")

if __name__ == "__main__":
    main()
