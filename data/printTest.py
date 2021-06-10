import numpy as np

import matplotlib.pyplot as plt
test=np.load('./loss_Bi-LSTMAttn_list.npy', allow_pickle=True)  #加载文件
print('shape: ', test.shape)
print('test: ', test)




plt.clf()
filename = ['bi_lstm_attn_model', 'bi_lstm_model', 'lstm_model']
color = ['red', 'blue', 'green', 'black']
loss_list = np.array(test)
plt.plot(np.arange(13359), loss_list, color=color[0], lw=2, label="loss")
plt.xlabel('Batch Numbers')
plt.ylabel('Loss')
plt.ylim([0, 10])
plt.xlim([0, 500])
plt.title('Loss per Batch')
plt.legend(loc="upper right")
plt.grid(True)
# plt.savefig('../image/loss')
plt.savefig('loss.eps', format='eps')
plt.show()

