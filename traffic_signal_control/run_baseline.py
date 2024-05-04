import argparse
import pickle

import numpy as np
from moss import Engine, TlPolicy
from tqdm import tqdm


class Recorder:
    """
    This class can be used to record simulation results.    
    """

    def __init__(self, eng: Engine, enable=True):
        self.enable = enable
        if not enable:
            return
        self.eng = eng
        self.data = {
            'lanes': eng.get_lane_geoms(),
            'steps': []
        }

    def record(self):
        if not self.enable:
            return
        self.data['steps'].append({
            'veh_positions': self.eng.get_vehicle_positions(),
            'lane_states': self.eng.get_lane_statuses(),
        })

    def save(self, filepath):
        if not self.enable:
            return
        pickle.dump(self.data, open(filepath, 'wb'))


class ControllerBase:
    def __init__(self, eng: Engine):
        pass

    def step(self):
        pass


class FixedTime(ControllerBase):
    """
    Fixed time traffic signal control
    """

    def __init__(self, eng: Engine):
        self.eng = eng

    def step(self):
        for i, (j, k) in enumerate(zip(
            self.eng.get_junction_phase_ids(),
            self.eng.get_junction_phase_counts()
        )):
            if k:
                self.eng.set_tl_phase(i, (j + 1) % k)


class MaxPressure(ControllerBase):
    """
    Max pressure traffic signal control
    """

    def __init__(self, eng: Engine):
        self.eng = eng
        # save the lane information for each phase
        self.jpl = eng.get_junction_phase_lanes()

    def step(self):
        # get the waiting vehicle count of each lane
        P = self.eng.get_lane_waiting_vehicle_counts()
        for i, pl in enumerate(self.jpl):
            if pl:
                # calculate the pressure of each phase
                ps = [P[i].sum() - P[o].sum() for i, o in pl]
                # select the phase with maximal pressure
                self.eng.set_tl_phase(i, np.argmax(ps))


class SOTL(ControllerBase):
    """
    SOTL traffic signal control
    """

    def __init__(self, eng: Engine):
        self.eng = eng
        # save the lane information for each phase
        self.jpl = eng.get_junction_phase_lanes()

    def step(self):
        # get the waiting vehicle count of each lane
        P = self.eng.get_lane_waiting_vehicle_counts()
        for i, (p_id, p_cnt, pl) in enumerate(zip(
            self.eng.get_junction_phase_ids(),
            self.eng.get_junction_phase_counts(),
            self.jpl,
        )):
            # calculate the number of vehicles on GREEN lanes
            green_cnt = sum(P[i] for i in pl[p_id][0])
            # calculate the number of vehicles on RED lanes
            red_cnt = sum(P[i] for i in set(sum((i[0] for i in pl), []))) - green_cnt
            # if GREEN is too few and RED is too many, switch to the next phase
            if green_cnt < 10 and red_cnt > 30:
                self.eng.set_tl_phase(i, (p_id + 1) % p_cnt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/hangzhou')
    parser.add_argument('--algo', choices=['fixed_time', 'max_pressure', 'sotl', 'ft_builtin', 'mp_builtin'], default='ft_builtin')
    parser.add_argument('--steps', type=int, default=3600)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    print('Moss version:', Engine.__version__)
    print('TSC algo:', args.algo)

    # initialize the engine
    print('Loading data... ', end='')
    eng = Engine(
        map_file=f'{args.data}/map.bin',
        agent_file=f'{args.data}/agents.bin',
    )
    print('done.')

    # set all junctions to manual
    eng.set_tl_policy_batch(range(eng.junction_count), TlPolicy.MANUAL)

    # initialize recorder
    recorder = Recorder(eng, enable=args.save)

    controller = ControllerBase(eng)
    if args.algo == 'fixed_time':
        controller = FixedTime(eng)
    elif args.algo == 'max_pressure':
        controller = MaxPressure(eng)
    elif args.algo == 'sotl':
        controller = SOTL(eng)
    elif args.algo == 'ft_builtin':
        # use the builtin FixedTime algo in the engine
        eng.set_tl_duration_batch(range(eng.junction_count), args.interval)
        eng.set_tl_policy_batch(range(eng.junction_count),  TlPolicy.FIXED_TIME)
    elif args.algo == 'mp_builtin':
        # use the builtin MaxPressure algo in the engine
        # NOTE: the builtin one uses a different implementation than the python one above
        eng.set_tl_duration_batch(range(eng.junction_count), args.interval)
        eng.set_tl_policy_batch(range(eng.junction_count),  TlPolicy.MAX_PRESSURE)
    else:
        raise NotImplementedError

    with tqdm(range(args.steps), ncols=100) as bar:
        for step in bar:
            if (step+1) % args.interval == 0:
                controller.step()
            eng.next_step()
            recorder.record()
            bar.set_description(f'Running: {eng.get_running_vehicle_count()} Finished: {eng.get_finished_vehicle_count()} Avg_time: {eng.get_finished_vehicle_average_traveling_time():.1f}s')
    recorder.save('playback.pkl')


if __name__ == '__main__':
    main()
