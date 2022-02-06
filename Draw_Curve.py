# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 18:11:33 2020

@author: Hang Yu
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# os.system('python sort_simulator_dc.py')
# os.system('python sort_simulator_na.py')
# os.system('python sort_simulator_sw.py')
# os.system('python sort_simulator_q.py')

alpha = np.loadtxt('DQN+TAMER_5.txt') 
demo = np.loadtxt('DQN+DEMO_5.txt') 
ps = np.loadtxt('DQN+ps_5.txt') 
q = np.loadtxt('DQN.txt') 
# q = np.loadtxt('res_q.txt') 

# tamer_dc = np.loadtxt('TAMER_dc_m.txt') 
# tamer_na = np.loadtxt('TAMER_na_m.txt') 
# tamer_sw = np.loadtxt('TAMER_sw_m.txt') 

# tamer_org = np.loadtxt('TAMER_o.txt') 


# print(sum(ps))
# print(sum(rf))
# print(sum(q))
# print(sum(rf)/sum(ps))
# print(sum(rf)/sum(q))

x=[i+1 for i in range(500)]
# per_100=[per[0] for i in range(100)]
# per_75=[per[1] for i in range(100)]
# per_50=[per[2] for i in range(100)]

plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.dpi'] = 300 
plt.rcParams["figure.figsize"] = (16,9)
plt.figure()
# tick_spacing = 100
fig, ax = plt.subplots(1, 1)
# ax.xaxis.set_major_locator(ticker.MaxNLocator(tick_spacing))


plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
# 设置坐标标签字体大小
ax.set_xlabel(..., fontsize=35)
ax.set_ylabel(..., fontsize=35)

            
plt.xlabel("Episodes")
plt.ylabel("Rewards")
ps_dc,=ax.plot(x,alpha,label='ALPHA',linewidth=6, color='teal')
ps_na,=ax.plot(x,ps,linewidth=6,label='DQN Policy Shaping', color='peru')
ps_sw,=ax.plot(x,q,linewidth=6, label='DQN',color='royalblue')
ps_q,=ax.plot(x,demo,label='DQN DEMO',linewidth=6, color='deeppink')


# tamer_ps_dc,=ax.plot(x,tamer_dc,label='TAMER-STEADY',linewidth=8, color='tomato')
# tamer_ps_na,=ax.plot(x,tamer_na,label='TAMER-Naive',linewidth=8, color='orange')
# tamer_ps_sw,=ax.plot(x,tamer_sw,label='TAMER-Window',linewidth=8, color='lightgreen')
# tamer_ps_org,=ax.plot(x,tamer_org,label='TAMER-Window',linewidth=8, color='red')


# ps_dc_d,=ax.plot(x[10],dc[10], marker='o',color='dodgerblue', markerfacecolor='dodgerblue', markersize=14)
# ps_na_d,=ax.plot(x[10],na[10], marker='^',color='dodgerblue', markerfacecolor='dodgerblue', markersize=14)
# ps_sw_d,=ax.plot(x[10],sw[10], marker='s',color='dodgerblue', markerfacecolor='dodgerblue', markersize=14)
# ps_q_d,=ax.plot(x[10],q[10], marker='h',color='dodgerblue', markerfacecolor='dodgerblue', markersize=14)

# tamer_ps_dc_d,=ax.plot(x[10],dc[10], marker='o',color='violet', markerfacecolor='violet', markersize=14)
# tamer_ps_na_d,=ax.plot(x[10],na[10], marker='^',color='violet', markerfacecolor='violet', markersize=14)
# tamer_ps_sw_d,=ax.plot(x[10],sw[10], marker='s',color='violet', markerfacecolor='violet', markersize=14)
# tamer_ps_org_d,=ax.plot(x[10],sw[10], marker='s',color='violet', markerfacecolor='violet', markersize=14)



# for i in range(100):
#     if i % 10 == 0:
#         ax.plot(x[i],dc[i], marker='o',color='dodgerblue', markerfacecolor='dodgerblue', markersize=14)
#         ax.plot(x[i],na[i], marker='^',color='dodgerblue', markerfacecolor='dodgerblue', markersize=14)
#         ax.plot(x[i],sw[i], marker='s',color='dodgerblue', markerfacecolor='dodgerblue', markersize=14)
#         ax.plot(x[i],q[i], marker='h',color='dodgerblue', markerfacecolor='dodgerblue', markersize=14)
#         ax.plot(x[i],tamer_dc[i], marker='o',color='violet', markerfacecolor='violet',markersize=14)
#         ax.plot(x[i],tamer_na[i], marker='^',color='violet', markerfacecolor='violet', markersize=14)
#         ax.plot(x[i],tamer_sw[i], marker='s',color='violet', markerfacecolor='violet', markersize=14)
#         ax.plot(x[i],tamer_org[i], marker='s',color='violet', markerfacecolor='violet', markersize=14)



# ax.plot(x,per_100,linewidth=2, color='gray')
# plt.annotate('100%', xy=(x[5],per_100[5]), xytext=(x[2],per_100[5]+100)) 
# ax.plot(x,per_75,linewidth=2, color='gray')
# plt.annotate('75%', xy=(x[5],per_75[5]), xytext=(x[2],per_75[5]+100)) 
# ax.plot(x,per_50,linewidth=2, color='gray')
# plt.annotate('50%', xy=(x[5],per_50[5]), xytext=(x[2],per_50[5]+100)) 



# for label in ax.get_xticklabels():
#      label.set_rotation(30)  # 旋转30度
#      label.set_horizontalalignment('right')  # 向右旋转
# plt.legend([(ps_dc,ps_dc_d),(ps_na,ps_na_d),(ps_sw,ps_sw_d),(ps_q,ps_q_d),(tamer_ps_dc,tamer_ps_dc_d),(tamer_ps_na,tamer_ps_na_d),(tamer_ps_sw,tamer_ps_sw_d),(tamer_ps_org,tamer_ps_org_d)],
#             ['PS-STEADY','PS-Naive','PS-Window','Q-Learning','TAMER-STEADY','TAMER-Naive','TAMER-Window','TAMER'],loc=[0,1],ncol=4,fontsize=25)

plt.legend(loc=[0,1],ncol=4,fontsize=25)

# plt.legend()
plt.savefig("T_Performance.png")