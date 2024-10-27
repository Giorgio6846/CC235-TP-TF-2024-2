import cv2
import os 
impo
depthFile = 

def showIMG():
    depthIMG = cv2.imread(
        os.path.join(os.path.dirname(__file__), "data", "DepthImage.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    cv2.imshow("image", depthIMG)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Read Redwood dataset")
color_raw = o3d.io.read_image("../../test_data/RGBD/color/00000.jpg")
depth_raw = o3d.io.read_image("../../test_data/RGBD/depth/00000.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw)
print(rgbd_image)


showIMG()