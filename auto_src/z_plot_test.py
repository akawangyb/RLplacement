# --------------------------------------------------
# 文件名: z_plot_test
# 创建时间: 2024/8/14 3:46
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import matplotlib.pyplot as plt

epochs = []
loss = []

with open('output.log', 'r', encoding='utf-8') as file:
    for line in file:
        if line.startswith("epoch"):
            epoch = int(line.split()[1])
            critic_loss = float(line.split()[4])
            epochs.append(epoch)
            loss.append(critic_loss)

plt.plot(epochs, loss, label='Critic Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Critic Loss Over Epochs')
plt.legend()
plt.show()
