import streamlit as st
import numpy as np
import cv2
import math
from PIL import Image, ImageColor
from face_detector import get_face_detector, find_faces, draw_faces
from face_landmarks import get_landmark_model, detect_marks
from head_pose_estimation import head_pose_points

# Set page configs
st.set_page_config(
    page_title="Head Pose Estimation",
    page_icon="./assets/icon_head.png",
    layout="centered"
)

# Title section
title = '<p style="text-align: center;font-size: 40px;font-weight: 550; "> Head Pose Estimation </p>'
st.markdown(title, unsafe_allow_html=True)

st.markdown(
    "This project creates head pose estimator that can tell where the head is facing in degrees "
    "using **Python** and **OpenCV**."
)


# ---------- Sidebar section ------------
# bounding box thickness
bbox_thickness = 3
# # bounding box color
# bbox_color = (0, 255, 0)

with st.sidebar:
    st.image("./assets/icon_head2.png", width=200)

    title = '<p style="font-size: 25px;font-weight: 550;">Head Pose Settings</p>'
    st.markdown(title, unsafe_allow_html=True)

    color1, color2 = st.columns(2)
    with color1:
        xbox_color = ImageColor.getcolor(
            str(st.color_picker(label="Angle X-axis Color", value="#FFFF80")), "RGB")

    with color2:
        ybox_color = ImageColor.getcolor(
            str(st.color_picker(label="Angle Y-axis Color", value="#80FFFF")), "RGB")
    # st.write('The current color is', bbox_color)

    bbox_thickness = st.slider("Bounding Box Thickness", min_value=1, max_value=20,
                               help="Sets the tickness of bounding boxes",
                               value=bbox_thickness)

    st.info("NOTE : The output of detection will depend on above paramters")

    # About the programmer
    st.markdown("## Made by **Dwi Anggun**")
    st.write("Find me on "
             "[*linkedin.com/dwianggun*](https://www.linkedin.com/in/dwi-anggun-cahyati-jamil-251288207/)")


# ----------- Image Upload Section -----------
uploaded_file = st.file_uploader(
    "Choose a file (Only PNG & JPG images allowed)", type=['png', 'jpg'])

face_model = get_face_detector()
landmark_model = get_landmark_model()

if uploaded_file is not None:
    with st.spinner("Estimating Head Pose.."):
        image = Image.open(uploaded_file)

        # To convert PIL Image to numpy array:
        img = np.array(image)
        size = img.shape
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            # Right eye right corne
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # Camera internals
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype="double")

        # Detect faces
        faces = find_faces(img, face_model)

        if len(faces) == 0:
            st.warning(
                "No face detected. Make sure your face is visible in the camera with proper lighting."
            )
        else:
            # draw face
            facedraw = draw_faces(img, faces)
            
            i = 1
            for face in faces:
                marks = detect_marks(img, landmark_model, face)
                # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
                image_points = np.array([
                    marks[30],     # Nose tip
                    marks[8],     # Chin
                    marks[36],     # Left eye left corner
                    marks[45],     # Right eye right corne
                    marks[48],     # Left Mouth corner
                    marks[54]      # Right mouth corner
                ], dtype="double")
                dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(
                    model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

                # Project a 3D point (0, 0, 1000.0) onto the image plane.
                # We use this to draw a line sticking out of the nose

                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array(
                    [(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

                # for p in image_points:
                #    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]),
                      int(nose_end_point2D[0][0][1]))
                x1, x2 = head_pose_points(img, rotation_vector,
                                          translation_vector, camera_matrix)

                #cv2.line(img, p1, p2, xbox_color, bbox_thickness)
                #cv2.line(img, tuple(x1), tuple(x2), ybox_color, bbox_thickness)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("Person {}:".format(i))

                with col2:
                    try:
                        m = (p2[1] - p1[1])/(p2[0] - p1[0])
                        ang1 = int(math.degrees(math.atan(m)))
                        st.write("X angle ", ang1)
                    except:
                        ang1 = 90

                with col3:
                    try:
                        m = (x2[1] - x1[1])/(x2[0] - x1[0])
                        ang2 = int(math.degrees(math.atan(-1/m)))
                        st.write("Y angle ", ang2)
                    except:
                        ang2 = 90

                i += 1

                # print('div by zero error')
                if ang1 >= 48:
                    st.write("Head down")
                    cv2.putText(img, 'Head down', (30, 30),
                                font, 2, ybox_color, bbox_thickness)
                elif ang1 <= -48:
                    st.write("Head up")
                    cv2.putText(img, 'Head up', (30, 30),
                                font, 2, ybox_color, bbox_thickness)

                if ang2 >= 48:
                    st.subheader("Person looking at the RIGHT")
                    cv2.putText(img, 'Head right', (30, 60),
                                font, 2, ybox_color, bbox_thickness)
                elif ang2 <= -48:
                    st.write("Head left")
                    cv2.putText(img, 'Head left', (90, 30),
                                font, 2, ybox_color, bbox_thickness)

                if -47 <= ang1 <= 47 & -47 <= ang2 <= 47:
                    print('Head forward')
                    cv2.putText(img, 'Head forward', (90, 30),
                                font, 2, ybox_color, bbox_thickness)

                cv2.putText(img, str(ang1), tuple(p1), font,
                            1, xbox_color, bbox_thickness)
                cv2.putText(img, str(ang2), tuple(x1), font,
                            1, ybox_color, bbox_thickness)

            # Display the output
            st.image(img)
