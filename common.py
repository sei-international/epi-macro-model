# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 18:34:09 2021

@author: Eric
"""
import datetime as dt

class Window:
    def __init__(self, start, end, ramp_up, ramp_down, effectiveness = 1.0):
        self.start = start
        self.end = end
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.effectiveness = effectiveness
    
    # Ramp to a maximum value of 1.0 and then back down
    # Ramp time can be zero
    def window(self, time):
        if time < self.start or time > self.end:
            w = 0
        elif time >= self.start + self.ramp_up and time <= self.end - self.ramp_down:
            w = 1
        elif time < self.start + self.ramp_up:
            w = (time - self.start)/self.ramp_up
        else:
            w = (self.end - time)/self.ramp_down
        return self.effectiveness * w

# This is a related set of functions that admits a dict with keys 'year', 'month', 'day' as argument
def get_datetime(d):
    return dt.date(d['year'],d['month'],d['day'])

def timesteps_between_dates(d0, d1, days_per_timestep = 1):
    return round((get_datetime(d1) - get_datetime(d0)).days/days_per_timestep)

def get_datetime_array(d0, d1, days_per_timestep = 1):
    tot_ts = timesteps_between_dates(d0, d1, days_per_timestep)
    return [get_datetime(d0) + dt.timedelta(days = x * days_per_timestep) for x in range(0, tot_ts)]

def timesteps_over_timedelta_weeks(td, days_per_timestep = 1):
    return round(dt.timedelta(weeks=td).days/days_per_timestep)
