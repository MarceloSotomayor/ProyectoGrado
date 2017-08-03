from Utilitarios import MPU6050
import time
import datetime
import os
i=0
muestreo=0.1
os.remove('mpu_log.txt')
print "INICIANDO"
f=open('mpu_log.txt','a')
f.write("FECHA"+","+"AC_EJEX"+","+"AC_EJEY"+","+"AC_EJEZ"+","+"GY_EJEX"+","+"GY_EJEY"+","+"GY_EJEZ"+"\n")
while(i<100):
    i=i+1
    imu = MPU6050()
    now=datetime.datetime.now()
    timestamp = now.strftime("%Y/%m/%d %H:%M") 
    (acc_x, acc_y, acc_z) = imu.get_acc()
    (gyro_x, gyro_y, gyro_z) = imu.get_gyro()
    outstring=str(timestamp)+","+str(acc_x)+","+str(acc_y)+","+str(acc_z)+","+str(gyro_x)+","+str(gyro_y)+","+str(gyro_z)+"\n"
    f.write(outstring)
    time.sleep(muestreo)
f.close()
print "FINALIZANDO"    
