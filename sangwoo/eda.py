import matplotlib.pyplot as plt
from collections import defaultdict
from read import read_receipts

def get_spending_by_dept(receipts):
    spending_by_dept = defaultdict(float)
    for receipt in receipts:
        dept_costs = {}
        for scan in receipt.scans:
            dept_costs[scan.department] = dept_costs.get(scan.department, 0) + scan.price
        for dept, cost in dept_costs.items():
            spending_by_dept[dept] += cost
    return spending_by_dept

def get_top_depts_and_others(spending_by_dept, top_n=5):
    sorted_depts = sorted(spending_by_dept.items(), key=lambda x: x[1], reverse=True)
    top_depts = sorted_depts[:top_n]
    others = sum(spending for dept, spending in sorted_depts[top_n:])
    top_depts.append(('Others', others))
    return top_depts

def plot_pie_chart(receipts, top_n=7):
    spending_by_dept = get_spending_by_dept(receipts)
    top_depts_and_others = get_top_depts_and_others(spending_by_dept, top_n)
    departments = [ dept for dept, spending in top_depts_and_others]
    spending = [spending for dept, spending in top_depts_and_others]

    fig, ax = plt.subplots()
    ax.pie(spending, labels=departments, autopct='%1.1f%%')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Categories of consumers based on their primary department')
    department_mapping = {
        'Bakery & Pastry': 1,
        'Beer & Wine': 2,
        'Books & Magazines': 3,
        'Candy & Chips': 4,
        'Care & Hygiene': 5,
        'Cereals & Spreads': 6,
        'Cheese & Tapas': 7,
        'Dairy & Eggs': 8,
        'Freezer': 9,
        'Fruit & Vegetables': 10,
        'Household & Pet': 11,
        'Meat & Fish': 12,
        'Pasta & Rice': 13,
        'Salads & Meals': 14,
        'Sauces & Spices': 15,
        'Soda & Juices': 16,
        'Special Diet': 17,
        'Vegetarian & Vegan': 18
    }
    labels = [f"{department_mapping.get(dept, dept)}: {dept}" for dept in departments]

    plt.show()


# Example usage
receipts = read_receipts('supermarket.csv')
plot_pie_chart(receipts)