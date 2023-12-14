import cv2
import mediapipe as mp
import random
from collections import deque
import statistics as st
import csv

def calculate_winner(cpu_choice, player_choice):

    # Determines the winner of each round when passed the computer's and player's moves

    if player_choice == "Invalid":
        return "Invalid!"

    if player_choice == cpu_choice:
        return "Tie!"

    elif player_choice == "Rock" and cpu_choice == "Scissors":
        return "You win!"

    elif player_choice == "Rock" and cpu_choice == "Paper":
        return "CPU wins!"

    elif player_choice == "Scissors" and cpu_choice == "Rock":
        return "CPU wins!"

    elif player_choice == "Scissors" and cpu_choice == "Paper":
        return "You win!"

    elif player_choice == "Paper" and cpu_choice == "Rock":
        return "You win!"

    elif player_choice == "Paper" and cpu_choice == "Scissors":
        return "CPU wins!"


def compute_fingers(hand_landmarks, count):

    # Coordinates are used to determine whether a finger is being held up or not
    # This is done by detemining whether the tip of the finger is above or below the base of the finger
    # For the thumb it determines whether the tip is to the left or right (depending on whether it's their right or left hand)

    # Index Finger
    if hand_landmarks[8][2] < hand_landmarks[6][2]:
        count += 1

    # Middle Finger
    if hand_landmarks[12][2] < hand_landmarks[10][2]:
        count += 1

    # Ring Finger
    if hand_landmarks[16][2] < hand_landmarks[14][2]:
        count += 1

    # Pinky Finger
    if hand_landmarks[20][2] < hand_landmarks[18][2]:
        count += 1

    # Thumb
    if hand_landmarks[4][3] == "Left" and hand_landmarks[4][1] > hand_landmarks[3][1]:
        count += 1
    elif hand_landmarks[4][3] == "Right" and hand_landmarks[4][1] < hand_landmarks[3][1]:
        count += 1
    return count


# Loading in from mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Using OpenCV to capture from the webcam
webcam = cv2.VideoCapture(0)

cpu_choices = ["Rock", "Paper", "Scissors"]
cpu_choice = "Nothing"
cpu_score, player_score = 0, 0
winner_colour = (0, 255, 0)
player_choice = "Nothing"
hand_valid = False
display_values = ["Rock", "Invalid", "Scissors", "Invalid", "Invalid", "Paper"]
winner = "None"
de = deque(['Nothing'] * 5, maxlen=5)

webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set width
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set height

cv2.namedWindow('Rock, Paper, Scissors', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Rock, Paper, Scissors', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

accumulated_hand_landmarks = []

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while webcam.isOpened():
        success, image = webcam.read()
        if not success:
            print("Camera isn't working")
            continue
        
        # Calculate center coordinates
        height, width, _ = image.shape
        center_x, center_y = int(width / 2), int(height / 2)

        # Draw a rectangle at the center
        rect_width, rect_height = 400, 400  # Adjust these values as needed
        top_left = (center_x - int(rect_width / 2), center_y - int(rect_height / 2))
        bottom_right = (center_x + int(rect_width / 2), center_y + int(rect_height / 2))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)


        image = cv2.flip(image, 1)

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        handNumber = 0
        hand_landmarks = []
        isCounting = False
        count = 0
        
        hand_landmarks_in_box = [] 

        # If at least one hand is detected this will execute.
        if results.multi_hand_landmarks:
            isCounting = True
            
            frame_hand_landmarks = []

            # hand_valid acts as a flag so that the CPU does not "play" a move multiple times.
            if player_choice != "Nothing" and not hand_valid:

                hand_valid = True
                cpu_choice = random.choice(cpu_choices)
                winner = calculate_winner(cpu_choice, player_choice)

                # Incrementing scores of player or CPU
                if winner == "You win!":
                    player_score += 1
                    winner_colour = (255, 0, 0)
                elif winner == "CPU wins!":
                    cpu_score += 1
                    winner_colour = (0, 0, 255)
                elif winner == "Invalid!" or winner == "Tie!":
                    winner_colour = (0, 255, 0)

            # Drawing the hand skeletons
            for hand in results.multi_hand_landmarks:
                hand_landmarks = []
                mp_drawing.draw_landmarks(
                    image,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Figures out whether it's a left hand or right hand in frame
                label = results.multi_handedness[handNumber].classification[0].label

                # Converts unit-less hand landmarks into pixel counts
                for id, landmark in enumerate(hand.landmark):
                    imgH, imgW, imgC = image.shape
                    xPos, yPos = int(landmark.x * imgW), int(landmark.y * imgH)

                    hand_landmarks.append([id, xPos, yPos, label])

                # Number of fingers held up are counted.
                count = compute_fingers(hand_landmarks, count)

                handNumber += 1
                

            #accumulate the hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                single_hand_landmark = []

                for idx, landmark in enumerate(hand_landmarks.landmark):
                    single_landmark = [
                        idx,
                        landmark.x,
                        landmark.y,
                        landmark.z if landmark.HasField('z') else None
                    ]
                    
                    if top_left[0] < xPos < bottom_right[0] and top_left[1] < yPos < bottom_right[1]:
                        single_hand_landmark.extend(single_landmark)
                        

                frame_hand_landmarks.append(single_hand_landmark)
                
                if top_left[0] < xPos < bottom_right[0] and top_left[1] < yPos < bottom_right[1]:
                    frame_hand_landmarks.append([player_choice])
                
            accumulated_hand_landmarks.append(frame_hand_landmarks)
            
            
            
            
            
            # for hand_landmarks in results.multi_hand_landmarks:
            #     landmarks_in_box = []  # Temporary list to store landmarks within the box for this hand

            #     for idx, landmark in enumerate(hand_landmarks.landmark):
            #         # Convert unit-less hand landmarks into pixel counts
            #         xPos, yPos = float(landmark.x * width), float(landmark.y * height)
            #         zPos = landmark.z if landmark.HasField('z') else None

            #         # Check if the landmark is within the bounding box
            #         if top_left[0] < xPos < bottom_right[0] and top_left[1] < yPos < bottom_right[1]:
            #             landmarks_in_box.append([idx, xPos, yPos, zPos])

            #     # Add the landmarks within the box for this hand to the list
            #     if len(landmarks_in_box) > 0:
            #         hand_landmarks_in_box.append(landmarks_in_box)
                    
            # accumulated_hand_landmarks.append(hand_landmarks_in_box)


        else:
            hand_valid = False

        # The number of fingers being held up is used to determine which move is made by the player
        if isCounting and count <= 5:
            player_choice = display_values[count]
        elif isCounting:
            player_choice = "Invalid"
        else:
            player_choice = "Nothing"

        # Adding the detected move to the left of the double-ended queue
        de.appendleft(player_choice)

        # Instead of using the first move detected, the mode is taken so that it provides a more reliable move detection.
        try:
            player_choice = st.mode(de)
        except st.StatisticsError:
            print("Stats Error")
            continue

        # Overlaying text on the webcam input to convey the score, move and round winner.
        cv2.putText(image, "You", (90, 75),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 5)

        cv2.putText(image, "CPU", (1050, 75),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)

        cv2.putText(image, player_choice, (45, 375),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 5)

        cv2.putText(image, cpu_choice, (1000, 375),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)

        cv2.putText(image, winner, (530, 650),
                    cv2.FONT_HERSHEY_DUPLEX, 2, winner_colour, 5)

        cv2.putText(image, str(player_score), (145, 200),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 5)

        cv2.putText(image, str(cpu_score), (1100, 200),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)

        cv2.imshow('Rock, Paper, Scissors', image)

        # Allows for the program to be closed by pressing the Escape key
        if cv2.waitKey(1) & 0xFF == 27:
            break


csv_filename = "hand_landmarks.csv"

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ["Hand" + str(i) + "_" + str(j) for i in range(2) for j in range(21)]
    header.append("Player_Choice")
    writer.writerow(header)

    for frame_landmarks in accumulated_hand_landmarks:
        frame_row = [data for sublist in frame_landmarks for data in sublist]
        writer.writerow(frame_row)

webcam.release()
cv2.destroyAllWindows()
