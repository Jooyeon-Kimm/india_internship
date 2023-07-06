import mediapipe as mp
import pandas as pd
import numpy as np
import cv2
import argparse
import math

def calculate_angle(pnt1, pnt2, pnt3):
    # Modify to np.array
    pnt1 = np.array(pnt1); 
    pnt2 = np.array(pnt2); 
    pnt3 = np.array(pnt3)  

    # Measure angles using tangents
    radians = np.arctan2(pnt3[1] - pnt2[1], pnt3[0] - pnt2[0]) - np.arctan2(pnt1[1] - pnt2[1], pnt1[0] - pnt2[0])
    
    # Find out angle and transforms within 180 degrees
    angle = np.abs(radians * 180.0 / math.pi)
    angle = angle if angle <= 180.0 else 360 - angle

    return angle

def detect_body(features, body_segment_name):
    return [
        features[mp.solutions.pose.PoseLandmark[body_segment_name].value].x,
        features[mp.solutions.pose.PoseLandmark[body_segment_name].value].y,
        features[mp.solutions.pose.PoseLandmark[body_segment_name].value].visibility
    ]

def detect_body_segments(features):
    body_segments = pd.DataFrame(columns=["body_segment", "x", "y"])

    for i, ftr in enumerate(mp.solutions.pose.PoseLandmark):
        ftr = str(ftr).split(".")[1]
        cord = detect_body(features, ftr)
        body_segments.loc[i] = ftr, cord[0], cord[1]

    return body_segments

def score_table(workout, display , counter, mode):
    cv2.putText(display, "Activity : " + workout.replace("-", " "),
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                cv2.LINE_AA)
    cv2.putText(display, "Counter : " + str(counter), (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(display, "Status : " + str(mode), (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return display

class BodySegmentAngle:
    def __init__(self, features):
        self.features = features

    def angle_of_the_left_arm_posture(self):
        left_shoulder = detect_body(self.features, "LEFT_SHOULDER")
        left_elbow = detect_body(self.features, "LEFT_ELBOW")
        left_wrist = detect_body(self.features, "LEFT_WRIST")
        return calculate_angle(left_shoulder, left_elbow, left_wrist)
    
    def angle_of_the_right_arm_posture(self):
        right_shoulder = detect_body(self.features, "RIGHT_SHOULDER")
        right_elbow = detect_body(self.features, "RIGHT_ELBOW")
        right_wrist = detect_body(self.features, "RIGHT_WRIST")
        return calculate_angle(right_shoulder, right_elbow, right_wrist)

    def angle_of_the_left_leg_posture(self):
        left_hip = detect_body(self.features, "LEFT_HIP")
        left_knee = detect_body(self.features, "LEFT_KNEE")
        left_ankle = detect_body(self.features, "LEFT_ANKLE")
        return calculate_angle(left_hip, left_knee, left_ankle)

    def angle_of_the_right_leg_posture(self):
        right_hip = detect_body(self.features, "RIGHT_HIP")
        right_knee = detect_body(self.features, "RIGHT_KNEE")
        right_ankle = detect_body(self.features, "RIGHT_ANKLE")
        return calculate_angle(right_hip, right_knee, right_ankle)

    def angle_of_the_neck_posture(self):
        right_shoulder = detect_body(self.features, "RIGHT_SHOULDER")
        left_shoulder = detect_body(self.features, "LEFT_SHOULDER")
        right_mouth = detect_body(self.features, "MOUTH_RIGHT")
        left_mouth = detect_body(self.features, "MOUTH_LEFT")
        right_hip = detect_body(self.features, "RIGHT_HIP")
        left_hip = detect_body(self.features, "LEFT_HIP")

        shoulder_average = [(right_shoulder[0] + left_shoulder[0]) / 2,
                        (right_shoulder[1] + left_shoulder[1]) / 2]
        mouth_average = [(right_mouth[0] + left_mouth[0]) / 2,
                     (right_mouth[1] + left_mouth[1]) / 2]
        hip_average = [(right_hip[0] + left_hip[0]) / 2, (right_hip[1] + left_hip[1]) / 2]

        return abs(180 - calculate_angle(mouth_average, shoulder_average, hip_average))

    def angle_of_the_abdomen_posture(self):
        # calculate angle of the avg shoulder
        right_shoulder = detect_body(self.features, "RIGHT_SHOULDER")
        left_shoulder = detect_body(self.features, "LEFT_SHOULDER")
        shoulder_average = [(right_shoulder[0] + left_shoulder[0]) / 2,
                        (right_shoulder[1] + left_shoulder[1]) / 2]

        # calculate angle of the avg hip
        right_hip = detect_body(self.features, "RIGHT_HIP")
        left_hip = detect_body(self.features, "LEFT_HIP")
        hip_average = [(right_hip[0] + left_hip[0]) / 2, (right_hip[1] + left_hip[1]) / 2]

        # calculate angle of the avg knee
        right_knee = detect_body(self.features, "RIGHT_KNEE")
        left_knee = detect_body(self.features, "LEFT_KNEE")
        knee_average = [(right_knee[0] + left_knee[0]) / 2, (right_knee[1] + left_knee[1]) / 2]

        return calculate_angle(shoulder_average, hip_average, knee_average)

# This code uses angles to get a count. This can then be used for posture correction. However, it has a weakness that makes it somewhat inaccurate. 
# We're also thinking about evolving it into an artificial intelligence.
class ExerciseType(BodySegmentAngle):
    def __init__(self, features):
        super().__init__(features)

    def push_up(self, counter, mode):
        left_arm_angle = self.angle_of_the_left_arm_posture()
        right_arm_angle = self.angle_of_the_left_arm_posture()
        avg_arm_angle = (left_arm_angle + right_arm_angle) // 2

        if mode:
            if avg_arm_angle < 70:
                counter += 1
                mode = False
        else:
            if avg_arm_angle > 160:
                mode = True

        return [counter, mode]


    def pull_up(self, counter, mode):
        nose = detect_body(self.features, "NOSE")
        left_elbow = detect_body(self.features, "LEFT_ELBOW")
        right_elbow = detect_body(self.features, "RIGHT_ELBOW")
        average_shoulder_y = (left_elbow[1] + right_elbow[1]) / 2

        if mode:
            if nose[1] > average_shoulder_y:
                counter += 1
                mode = False

        else:
            if nose[1] < average_shoulder_y:
                mode = True

        return [counter, mode]

    def squat(self, counter, mode):
        left_leg_angle = self.angle_of_the_right_leg_posture()
        right_leg_angle = self.angle_of_the_left_leg_posture()
        average_leg_angle = (left_leg_angle + right_leg_angle) // 2

        if mode:
            if average_leg_angle < 70:
                counter += 1
                mode = False
        else:
            if average_leg_angle > 160:
                mode = True

        return [counter, mode]

    def walk(self, counter, mode):
        right_knee = detect_body(self.features, "RIGHT_KNEE")
        left_knee = detect_body(self.features, "LEFT_KNEE")

        if mode:
            if left_knee[0] > right_knee[0]:
                counter += 1
                mode = False

        else:
            if left_knee[0] < right_knee[0]:
                counter += 1
                mode = True

        return [counter, mode]

    def sit_up(self, counter, mode):
        angle = self.angle_of_the_abdomen_posture()
        if mode:
            if angle < 55:
                counter += 1
                mode = False
        else:
            if angle > 105:
                mode = True

        return [counter, mode]

    def calculate_exercise(self, workout_type, counter, mode):
        if workout_type == "push-up":
            counter, mode = ExerciseType(self.features).push_up(
                counter, mode)
        elif workout_type == "pull-up":
            counter, mode = ExerciseType(self.features).pull_up(
                counter, mode)
        elif workout_type == "squat":
            counter, mode = ExerciseType(self.features).squat(
                counter, mode)
        elif workout_type == "walk":
            counter, mode = ExerciseType(self.features).walk(
                counter, mode)
        elif workout_type == "sit-up":
            counter, mode = ExerciseType(self.features).sit_up(
                counter, mode)
        else: 
            raise ValueError('This exercise has not yet been added to the function.')

        return [counter, mode]
    
def run():
    parser = argparse.ArgumentParser(description="AI_FITNESS")
    parser.add_argument("--exercise-type", type=str, required=True, help='What kind of exercise?')
    
    args = vars(parser.parse_args())

    media_pipe_drawing = mp.solutions.drawing_utils
    media_pipe_pose = mp.solutions.pose
    
    capture = cv2.VideoCapture(0)  # webcam
    capture.set(3, 800)  # width
    capture.set(4, 480)  # height

    with media_pipe_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        counter = 0  # movement of exercise
        mode = True  # state of move
        while capture.isOpened():
            success, display = capture.read()
            # result_screen = np.zeros((250, 400, 3), np.uint8)

            display = cv2.resize(display, (800, 480), interpolation=cv2.INTER_AREA)
            # recolor frame to RGB
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            display.flags.writeable = False
            # make detection
            results = pose.process(display)
            # recolor back to BGR
            display.flags.writeable = True
            display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)

            try:
                features = results.pose_landmarks.landmark
                counter, mode = ExerciseType(features).calculate_exercise(args["workout_type"], counter, mode)
            except:
                pass

            display = score_table(args["workout_type"], display, counter, mode)

            # render detections (for landmarks)
            media_pipe_drawing.draw_landmarks(
                display,
                results.pose_landmarks,
                media_pipe_pose.POSE_CONNECTIONS,
                media_pipe_drawing.DrawingSpec(color=(255, 255, 255),
                                    thickness=2,
                                    circle_radius=2),
                media_pipe_drawing.DrawingSpec(color=(174, 139, 45),
                                    thickness=2,
                                    circle_radius=2),
            )

            cv2.imshow('Video', display)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
