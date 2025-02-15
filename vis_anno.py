from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 图片和边界框信息
image_name = '0_Hemp_stamen_1_286.jpg'
bbox_x = 1207
bbox_y = 837
bbox_width = 1236
bbox_height = 1267

# 图片路径
image_directory = r'E:\Gong\new\Grounding-Dino-FineTuning-main\stamen_dataset\images\val'
image_path = f"{image_directory}\\{image_name}"

# 可视化图片和边界框
def visualize_image_with_bbox(image_path, bbox_x, bbox_y, bbox_width, bbox_height):
    # 加载图片
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')  # 不显示坐标轴

    # 绘制边界框
    rect = patches.Rectangle((bbox_x, bbox_y), bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)

    # 显示图片
    plt.show()

# 调用可视化函数
visualize_image_with_bbox(image_path, bbox_x, bbox_y, bbox_width, bbox_height)