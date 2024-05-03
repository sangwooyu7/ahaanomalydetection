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

    def biggest_spending_department(self):
        dept_costs = {}
        for scan in self.scans:
            dept_costs[scan.department] = dept_costs.get(scan.department, 0) + scan.price
        return max(dept_costs.items(), key=lambda x: x[1])[0] if dept_costs else None

    def time_scans_ratio(self):
        return self.total_time / len(self.scans) if self.scans else 0

    def time_cost_ratio(self):
        return self.total_time / self.total_cost if self.total_cost > 0 else 0

    def cost_scans_ratio(self):
        return self.total_cost / len(self.scans) if self.total_cost > 0 else 0

    def number_of_scans(self):
        return len(self.scans)