import json

data = json.load(open('label_work/priority_2.json'))
print(f"Total: {len(data)}")
for i, item in enumerate(data):
    print(f"{i+1}. {item['code']} | {item['name']} | curr: {item.get('current_business', {}).get('primary')}")
