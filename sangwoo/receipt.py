class Receipt:
    def __init__(self):
        self.reasons = []
        self.scans = []
        self.total_time = 0
        self.total_cost = 0
        self.sus = False

    def add_scan(self, scan):
        self.scans.append(scan)
        self.total_time += scan.time
        self.total_cost += scan.price

    def get_scans(self):
        scans_data = []
        for scan in self.scans:
            scans_data.append([scan.department, scan.time, scan.price])
        return scans_data

    def calculate_time_variance(self):
        if len(self.scans) < 2:
            return 0  # Variance requires at least two data points
        times = [scan.time for scan in self.scans]
        intervals = [t - s for s, t in zip(times[:-1], times[1:])]
        mean_interval = sum(intervals) / len(intervals)
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        return variance

    def flag_as_sus(self, reason):
        self.sus = True
        self.reasons.append(reason)
        