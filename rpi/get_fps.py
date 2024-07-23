import pandas as pd
import matplotlib.pylab as plt


def read_pts(path):
    colnames=['Timestamp']
    df=pd.read_csv(path+'timestamp.txt',names=colnames,header=None,index_col=False)
    df['diff']=df['Timestamp'].diff()
    df['diff']=df['diff']/1000
    df['fps']=1/df['diff']
    df['timestamp']=df['Timestamp']/1000
    df=df.fillna(0)
    df['sec']=[int(str(i).split('.')[0]) for i in df['timestamp']]
    return df

def plot_fps(df_list,ti,path):
    for df in df_list:
        df['fps'][1:].plot(label=f'Actual FPS: {round(df["fps"][1:].mean(),2)} +/- {round(df["fps"][1:].std(),4)} ')
        plt.title(f'Actual FPS: {round(df["fps"][1:].mean(),2)} +/- {round(df["fps"][1:].std(),4)} ',
                fontweight = 'bold',size =15,pad=10)
    plt.ylim(50,60)
    plt.xlabel('Frame',fontweight='bold',size=15)
    plt.ylabel('FPS',fontweight='bold',size=15)
    plt.title(ti, fontweight = 'bold', size=20)
    plt.legend()
    plt.savefig(f'{path}fps_plot.svg')
#direc='/home/pi/temp/ed2_hbag16_2023-01-09_1/'
#paths=[direc+i+'/' for i in os.listdir(direc)]
#for path in paths:
path='/media/pi/pi_img2/FG_test_2024-02-03_1/2024-02-03_21-32-24_0/'
df=read_pts(path)
plot_fps([df],'tat',path)