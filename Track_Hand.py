import cv2
import mediapipe as mp

class HandDetector():
    def __init__(self, mode=False, n_hands=2, complexity=1, min_detect_confidence=0.5, min_track_confidence=0.5):

        self.mode = mode
        self.n_hands = n_hands
        self.complexity = complexity
        self.min_detect_confidence = min_detect_confidence
        self.min_track_confidence = min_track_confidence
        
        # create a object of hand class
        self.mphands = mp.solutions.hands    
        # Hands object
        self.hands = self.mphands.Hands(
                                        self.mode,
                                        self.n_hands,
                                        self.complexity,
                                        self.min_detect_confidence ,
                                        self.min_track_confidence
                                        )     

        # for drawing lines and landmark points on hand 
        self.mp_draw = mp.solutions.drawing_utils



    def findHands(self, img):
        # convert image to rgb 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # process the image and store the information in result
        result = self.hands.process(imgRGB)   # rgb image must be passed

        if result.multi_hand_landmarks:         # it contains landmarks for all hands
            for one_hand_landmarks in result.multi_hand_landmarks: 
                # one_hand_landmarks  ->  draws the landmark points on hand
                # mphands.HAND_CONNECTIONS -> draw connecting lines on hands 
                self.mp_draw.draw_landmarks(img, one_hand_landmarks, self.mphands.HAND_CONNECTIONS)  


                
    def findPos(self, img, hand_no=0, draw=True):
        lm_list = []
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(imgRGB) 

        if result.multi_hand_landmarks:         # it contains landmarks for all hands
            hand = result.multi_hand_landmarks[hand_no]

            if draw:
                self.mp_draw.draw_landmarks(img, hand, self.mphands.HAND_CONNECTIONS)  

            # all landmarks for current hand 
            for id, pos in enumerate(hand.landmark):
                # id -> landmark no
                # pos -> coordinates of this particular landmark   ( in decimal, multiply by H, W to get coordinates)
                H, W, C = img.shape
                x = int(pos.x * W) 
                y = int(pos.y * H)
                lm_list.append([x, y])
        
        return lm_list
