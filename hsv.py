import cv2
import matplotlib.pyplot as plt

# hover over image and get hsv values

# Load image
image = cv2.imread("14/obj/frame_000734.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

fig, ax = plt.subplots()
ax.imshow(image_rgb)
ax.set_title("Hover over the image to see HSV values (OpenCV ranges)")
ax.axis("off")

# Add a text annotation for HSV values
text = ax.text(10, 30, "", color="white", fontsize=12,
               bbox=dict(facecolor="black", alpha=0.5))

def format_hsv(event):
    if event.inaxes == ax:
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= x < hsv_image.shape[1] and 0 <= y < hsv_image.shape[0]:
            h, s, v = hsv_image[y, x]
            text.set_text(f"({x},{y}) → H={h}, S={s}, V={v}")
            fig.canvas.draw_idle()

# Connect hover event
fig.canvas.mpl_connect("motion_notify_event", format_hsv)

plt.show()
