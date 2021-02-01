import numpy as np 
import pandas as pd
import datetime
class environment(object):
    def __init__(self, stock, date, time_step):
        self.mode = 'train'
        self.day  = 0
        self.time_step = time_step
        self.eval_data  = None
        self.train_data = None
        self.get_data(stock, date)
    
    def get_data(self, stock, date):
        df = pd.read_csv('./financial/data/' + stock)
        df.index = pd.to_datetime(df.Ntime.values, format = '%Y%m%d')
        df.Ntime = df.index

        test_end_time    = self.get_next_timestep(date, month = 3)
        test_start_time  = date
        train_end_time   = self.get_next_timestep(date, day = -1)
        train_start_time = self.get_next_timestep(train_end_time, year = -1) 
        
        t = str(df[train_start_time:].index[0])[:10]
        
        train_start = str(df[:t].iloc[-self.time_step,:].Ntime)[:10]
        
        t = str(df[test_start_time:].index[0])[:10]
        test_start = str(df[:t].iloc[-self.time_step].Ntime)[:10]
        


        self.train_data = df[train_start: train_end_time].iloc[:, 1:].values
        self.test_data = df[test_start: test_end_time].iloc[:, 1:].values

    def reset(self):
        self.day = 0
        return self.get_observation()
    def step(self, action):
        self.day += 1

        reward = self.get_reward(action)
        is_done = self.finish()
        next_state = None if is_done else self.get_observation()
        
        return next_state, reward, is_done

    def get_reward(self, action):
        return 1

    def finish(self):
        if self.mode is 'train':
            if self.day > len(self.train_data):
                return True
            else:
                return False
        elif self.mode is 'test':
            if self.day > len(self.test_data):
                return True
            else:
                return False

    def get_observation(self):
        if self.mode is 'train':
            return self.train_data[self.day:self.day + self.time_step, :].T.reshape(1,-1,self.time_step )
        elif self.mode is 'test':
            return self.test_data[self.day:self.day + self.time_step, :].T.reshape(1,-1,self.time_step )

    def get_next_timestep(self, date, year = 0, month = 0, day = 0):
        '''
        year, month, day can be positive or negative number 
        positive number return the date after  years, months or days
        negative number return the date before years, months or days
        '''
        y = int(date[:4])   + year
        m = int(date[5:7])  + month
        d = int(date[8:])
        sign = 1 if day >=0 else -1   
        day = abs(day)
        while(m > 12):
            y +=1
            m = m-12
        while(m <= 0):
            y -= 1
            m += 12

        if(m >= 10):
            m = str(m)
        else:
            m = '0'+str(m) 
        
        if(d>=10):
            d = str(d)
        else:
            d = '0'+str(d)

        date = datetime.datetime.strptime(str(y)+'-'+str(m)+'-'+str(d), '%Y-%m-%d')
        date = str(date + datetime.timedelta(days = day) * sign)[:10]
        return date

if __name__ == '__main__':
    env = environment(stock = 'S&P500', date = '2010-01-01', time_step = 10)
    env.get_observation()