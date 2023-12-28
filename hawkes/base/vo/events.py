import numpy as np

class Events:
    # NOTE: 生成はEventsFactoryを使用
    def __init__(self, events_grouped_by_mark, events_ordered_by_time, end_time: float):
        self._events_grouped_by_mark = events_grouped_by_mark
        self._events_ordered_by_time = events_ordered_by_time
        self._end_time = end_time
        self._dim = len(events_grouped_by_mark)

    @property
    def grouped_by_mark(self):
        return self._events_grouped_by_mark

    @property
    def ordered_by_time(self):
        return self._events_ordered_by_time

    @property
    def end_time(self) -> float:
        return self._end_time

    @property
    def dim(self) -> int:
        return self._dim

class EventsFactory:
    @staticmethod
    def from_events_grouped_by_mark(events, end_time: float) -> Events:
        if not isinstance(events, list):
            raise TypeError('インスタンス生成エラー: イベントリストはlistである必要があります')
        for event in events:
            if not isinstance(event, np.ndarray):
                raise TypeError('インスタンス生成エラー: イベントリストの各次元要素はnumpy.ndarrayである必要があります')
            for t in event:
                if not isinstance(t, float):
                    raise TypeError('インスタンス生成エラー: イベントの発生時刻はfloatである必要があります')

        ordered_events = EventsFactory._order_by_time(events)

        if ordered_events[-1][0] > end_time:
            raise ValueError('インスタンス生成エラー: 最後のイベントの発生時刻が終了時刻を超えています')

        return Events(events, ordered_events, end_time)

    @staticmethod
    def from_events_ordered_by_time(events, end_time: float) -> Events:
        if not isinstance(events, np.ndarray):
            raise TypeError('インスタンス生成エラー: イベントリストはnumpy.ndarrayである必要があります')
        for t, mark in events:
            if not isinstance(t, float):
                raise TypeError('インスタンス生成エラー: イベントの発生時刻はfloatである必要があります')
            if not isinstance(mark, int):
                raise TypeError('インスタンス生成エラー: イベントのマークはintである必要があります')

        if events[-1][0] > end_time:
            raise ValueError('インスタンス生成エラー: 最後のイベントの発生時刻が終了時刻を超えています')

        grouped_events = EventsFactory._group_by_mark(events)

        return Events(grouped_events, events, end_time)

    @staticmethod
    def _group_by_mark(events_ordered_by_time):
        num_dim = np.max(events_ordered_by_time[:, 1]) + 1
        events_group_by_dim = [None] * num_dim
        for dim in range(num_dim):
            events_group_by_dim[dim] = events_ordered_by_time[events_ordered_by_time[:, 1] == dim, 0]
        return events_group_by_dim

    @staticmethod
    def _order_by_time(events_grouped_by_mark):
        events, dims = zip(*[(event, dim) for dim, events_dim in enumerate(events_grouped_by_mark) for event in events_dim])
        events, dims = np.array(events), np.array(dims)
        sorted_indices = np.argsort(events)
        return np.column_stack((events[sorted_indices], dims[sorted_indices]))