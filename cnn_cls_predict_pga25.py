import time
import math
import numpy as np
import torch
import torch.nn.functional as F
import model_cnn_cls
import PyEW
from datetime import datetime, timedelta
from collections import deque
import threading
import pickle


class model_cnn():
    def __init__(self, model=None, param_path=None):
        self.model = model
        checkpoint = torch.load(param_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def output(self, input):
        input = np.expand_dims(input,axis=0)
        input = torch.from_numpy(input).float()
        with torch.no_grad():
            probs = torch.squeeze(F.softmax(self.model(input), dim=1)).numpy()

        return probs

    def output_max(self, input):
        input = np.expand_dims(input,axis=0)
        input = torch.from_numpy(input).float()
        with torch.no_grad():
            output = np.argmax(torch.squeeze(F.softmax(self.model(input), dim=1)).numpy())

        return output

# to save wave form from WAVE_RING
class WaveSaver(threading.Thread):
    def __init__(self, thread_name):
        super(WaveSaver, self).__init__(name=thread_name)
    
    def run(self):
        global waveform
        global MyModule

        while True:
            # get raw waveform from WAVE_RING
            wave = MyModule.get_wave(0)
            threeAxis = {}

            # filter no-data and not start from HLZ(because the order of a station is HLZ, HLN, HLE)
            if wave == {} or wave["channel"] != "HLZ":
                continue
            threeAxis["HLZ"] = wave

            # get three axis wave form at once
            while True:
                tmp_wave = MyModule.get_wave(0)
                if tmp_wave != {}:
                    threeAxis[tmp_wave["channel"]] = tmp_wave
                    if len(threeAxis) >= 3:
                        break

            # save three axis waveform and station info to a dictionary
            for (channel,axisWave) in threeAxis.items():
                station = axisWave['station']
                location = axisWave['location']
                if ((station + "_" + channel + "_" + location + "_data") in waveform):
                    if (axisWave['startt'] != waveform[station + "_" + channel + "_" + location + "_startt"][-1]):
                        waveform[station + "_" + channel + "_" + location + "_data"].append(axisWave['data'][:])
                        waveform[station + "_" + channel + "_" + location + "_startt"].append(axisWave['startt'])
                        waveform[station + "_" + channel + "_" + location + "_endt"].append(axisWave['endt'])
                else:
                    waveform[station + "_" + channel + "_" + location + "_data"] = deque(maxlen=30)
                    waveform[station + "_" + channel + "_" + location + "_startt"] = deque(maxlen=30)
                    waveform[station + "_" + channel + "_" + location + "_endt"] = deque(maxlen=30)
                    waveform[station + "_" + channel + "_" + location + "_data"].append(axisWave['data'][:])
                    waveform[station + "_" + channel + "_" + location + "_startt"].append(axisWave['startt'])
                    waveform[station + "_" + channel + "_" + location + "_endt"].append(axisWave['endt'])
            # print(wave["station"])

# to handle data when there is a new station be picked
class PickHandler(threading.Thread):
    def __init__(self, thread_name):
        super(PickHandler, self).__init__(name=thread_name)

    def run(self):
        global waveform
        global MyModule
        global mode_cnn
        global time_window

        # a waiting list to process if multiple stations comes into PICK_RING simultaneously
        wait_list = deque()

        # listen PICK_RING
        while True:
            log_msg = "============================"

            # get picked station in PICK_RING
            pick_msg = MyModule.get_bytes(1, 150)

            # if there's no data and waiting list is empty
            if pick_msg == (0, 0) and len(wait_list) == 0:
                continue

            # if get picked station, then get its info and add to waiting list
            if pick_msg != (0, 0):
                pick_str = pick_msg[2][:pick_msg[1]].decode("utf-8")
                print(datetime.now(), pick_str)
                log_msg += "\n[" + str(datetime.now()) + "] " + pick_str
                wait_list.append(pick_str)

            # get the first one data in waiting list
            pick_info = wait_list[0]
            station = pick_info.split()[0]
            pstime = float(pick_info.split()[-4])
            channel = pick_info.split()[1]
            network = pick_info.split()[2]
            location = pick_info.split()[3]
            observe_time = int(pick_info.split()[-1])

            # only use HLZ, 2 seconds oberserve time, location 01 to predict
            if channel != 'HLZ' or observe_time != 2 or location != '01':
                wait_list.popleft()
                continue

            # to handle data that not in waveform, it wave be deleted from wait_list
            if station + "_HLZ_" + location + "_startt" not in waveform:
                print("not in waveform")
                wait_list.popleft()
                continue
            # to handle initial condition that waveform is less than time_window
            if len(waveform[station + "_HLZ_" + location + "_data"]) < time_window:
                print("length less than " + str(time_window))
                continue

            # get all waves' start time of the station in WAVE_RING
            tmp_start = waveform[station + "_HLZ_" + location + "_startt"]

            # if the latest wave start time is greater than p arrival time, save it and predict it
            if tmp_start[-1] - pstime >= (time_window):
                hlz_data = waveform[station + "_HLZ_" + location + "_data"]
                hln_data = waveform[station + "_HLN_" + location + "_data"]
                hle_data = waveform[station + "_HLE_" + location + "_data"]

                hlz = np.array([], dtype=int)
                hln = np.array([], dtype=int)
                hle = np.array([], dtype=int)

                timeIndex = math.floor(pstime - tmp_start[0])
                if timeIndex < 0: timeIndex = 0

                # prepare for predicting PGA
                for i in range(timeIndex,timeIndex+time_window):
                    try:
                        hlz = np.concatenate((hlz, hlz_data[i]))
                        hln = np.concatenate((hln, hln_data[i]))
                        hle = np.concatenate((hle, hle_data[i]))
                    except TypeError:
                        print(station)
                    except ValueError:
                        print(station)

                # transfer raw data to accerate
                sqrtg = math.sqrt(9.8)
                hlz = (hlz / 980) / sqrtg
                hln = (hln / 980) / sqrtg
                hle = (hle / 980) / sqrtg

                # mean
                hlz = hlz - np.mean(hlz)
                hln = hln - np.mean(hln)
                hle = hle - np.mean(hle)

                inp = np.array([hln[:], hle[:], hlz[:]])
                startt_local = datetime.utcfromtimestamp(tmp_start[timeIndex]) + timedelta(hours=8)
                endt_local = datetime.utcfromtimestamp(tmp_start[timeIndex] + time_window) + timedelta(hours=8)

                # print result
                msg = '\n============================'
                msg += ('\nstation: ' + station)
                msg += ('\nstart: ' + startt_local.strftime('%Y-%m-%d %H:%M:%S'))
                msg += ('\nend: ' + endt_local.strftime('%Y-%m-%d %H:%M:%S'))
                # print('start predict:',datetime.now())
                log_msg += "\n[" + str(datetime.now()) + "] start predict"
                prob = mode_cnn.output(inp)
                # print('end predict:',datetime.now())
                log_msg += "\n[" + str(datetime.now()) + "] end predict"
                msg += ("\nProbability: " + str(prob))
                msg += ("\nPGA>25: ")
                if prob[0] <= prob[1]:
                    msg += "yes"
                    # put result to RING
                    MyModule.put_msg(2, 14, pick_info)
                else:
                    msg += "no"
                msg += '\n============================'
                # print(msg)
                log_msg += msg

                # write log to file
                with open("pick_info_pga25_log.txt","a") as pif:
                    pif.write(log_msg)
                    pif.close()
                wait_list.popleft()

                # save wave that model saw
                # with open("pick_info_pga25_wave.txt", "ab") as fp:
                #     staInfo = {
                #         station + "_" + str(pstime) + "_" + location: {
                #             'pick_info': pick_info,
                #             'HLZ_data': hlz,
                #             'HLN_data': hln,
                #             'HLE_data': hle,
                #             'start_time': tmp_start[timeIndex]
                #         }
                #     }
                #     bin_obj = pickle.dumps(staInfo)
                #     fp.write(bin_obj)
                #     fp.close()
                # print(wait_list)

time_window = 3
print("time window: " + str(time_window))

try:

    # to save all raw wave form data
    waveform = {}

    # connect to earthworm, add WAVE_RING and PICK_RING and an OUTPUT_RING
    MyModule = PyEW.EWModule(1000, 172, 255, 30.0, False)
    MyModule.add_ring(1038)     # WAVE_RING
    MyModule.add_ring(1005)     # PICK_RING
    MyModule.add_ring(1037)     # OUTPUT_RING

    # initialize and load model
    if time_window == 1:
        model = model_cnn_cls.CNN1()
    elif time_window == 2:
        model = model_cnn_cls.CNN2()
    elif time_window == 3:
        model = model_cnn_cls.CNN3()
    elif time_window == 4:
        model = model_cnn_cls.CNN4()
    else:
        model = model_cnn_cls.CNN5()
    mode_cnn = model_cnn(model=model, param_path=str(time_window)+"00_cwb+tsmip.ckpt")
    
    wave_count = 0
    # flush PICK_RING
    while MyModule.get_bytes(1, 150) != (0, 0):
        wave_count += 1
        continue
    print("PICK_RING flushed with " + str(wave_count) + " waves flushed")

    wave_count = 0
    # flush WAVE_RING
    while MyModule.get_wave(0) != {}:
        wave_count += 1
        continue
    print("WAVE_RING flushed with " + str(wave_count) + " waves flushed")

    wave_saver = WaveSaver('waveServer')
    wave_saver.start()

    pick_handler = PickHandler('pickHandler')
    pick_handler.start()
except KeyboardInterrupt:
    MyModule.goodbye()
