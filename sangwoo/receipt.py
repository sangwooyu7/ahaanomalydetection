import math
from collections import defaultdict

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
        sorted_dept_costs = sorted(dept_costs.items(), key=lambda x: x[1], reverse=True)
        if sorted_dept_costs:
            return sorted_dept_costs[0][0]
        else:
            return None

    def second_biggest_spending_department(self):
        dept_costs = {}
        for scan in self.scans:
            dept_costs[scan.department] = dept_costs.get(scan.department, 0) + scan.price
        sorted_dept_costs = sorted(dept_costs.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_dept_costs) >= 2:
            return sorted_dept_costs[1][0]
        else:
            return None

    def most_scanned_department(self):
        department_counts = {}
        for scan in self.scans:
            department_counts[scan.department] = department_counts.get(scan.department, 0) + 1

        if not department_counts:
            return None

        most_scanned_dept = max(department_counts.items(), key=lambda x: x[1])[0]
        return most_scanned_dept

    def time_scans_ratio(self):
        return self.total_time / len(self.scans) if self.scans else 0

    def time_cost_ratio(self):
        return self.total_time / self.total_cost if self.total_cost > 0 else 0

    def cost_scans_ratio(self):
        return self.total_cost / len(self.scans) if self.total_cost > 0 else 0

    def number_of_scans(self):
        return len(self.scans)

    def extract_features(self, min_confidence=0.5):
        log_total_time = math.log(max(self.total_time, 1) + 1)  # Avoid math domain error
        log_total_cost = math.log(max(self.total_cost, 1) + 1)  # Avoid math domain error
        num_scans = len(self.scans)
        time_scans_ratio = num_scans / self.total_time if self.total_time > 0 else 0
        time_cost_ratio = self.time_cost_ratio()
        cost_scans_ratio = self.total_cost / num_scans if num_scans > 0 else 0
        time_variance = self.calculate_time_variance()

        dept_costs = defaultdict(float)
        dept_scans = defaultdict(int)

        dept_changes = 0
        back_and_forth = 0
        visited_depts = set()
        prev_dept = None

        for scan in self.scans:
            dept_costs[scan.department] += scan.price
            dept_scans[scan.department] += 1
            if prev_dept is not None and scan.department != prev_dept:
                dept_changes += 1
                if scan.department in visited_depts:
                    back_and_forth += 1
                visited_depts.add(prev_dept)
            prev_dept = scan.department

        features = [
            time_variance,
            dept_changes,
            time_scans_ratio,
            cost_scans_ratio,
            time_cost_ratio
        ]

        return features