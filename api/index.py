from flask import Flask, request, jsonify
import ollama
from pydantic import BaseModel, Field
from typing import List, Literal

app = Flask(__name__)

class TimeSlot(BaseModel):
    day: Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    # ge means greater than or equal, le means less than or equal
    hour: int = Field(ge=9, le=16)

class UserInput(BaseModel):
    available_slots: List[TimeSlot]
    preferred_slots: List[TimeSlot]

# This function takes a user's message and returns a UserInput object, which contains the available and preferred slots.
def get_user_data(user_text: str):
    system_prompt = """
    You are a scheduling assistant. Standard Work Hours: 09:00 to 17:00 (5 PM).
    
    RULES FOR GENERATING SLOTS:
    1. Ranges: "9 to 11" means hours [9, 10]. Do not include the end hour.
    2. Explicit Days: If a user says "Free Tuesday", only list Tuesday slots.
    3. Negative Constraints: If a user says "Busy Friday", you MUST list ALL available hours for Monday, Tuesday, Wednesday, and Thursday, plus the free hours on Friday.
    4. IMPLIED AVAILABILITY: Unless a user explicitly excludes a day, assume they are available 09:00-17:00.
    """

    response = ollama.chat(
        model='llama3.2',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_text}
        ],
        format=UserInput.model_json_schema(), 
        options={'temperature': 0} # Set temp to 0 to make it more logical/strict
    )

    return UserInput.model_validate_json(response['message']['content'])

# This function finds the minimum common slots between users, and also a preferred schedule, through concept of set intersection.
def find_best_times(users_data: list[UserInput]):
    if not users_data: return []

    # Initializing the first user before schedule intersection
    first_user = users_data[0]
    # print(f"First User: {first_user}")
    common_slots = set()
    for slot in first_user.available_slots:
        common_slots.add((slot.day, slot.hour))

    # Intersecting with remaining users
    remaining_users = users_data[1:]
    for i, user in enumerate(remaining_users):
        current_user_slots = set()
        for slot in user.available_slots:
            current_user_slots.add((slot.day, slot.hour))
        
        # Debugging: See why intersection might fail
        # print(f"Intersecting with User {i+2}")
        
        # The Math
        common_slots = common_slots.intersection(current_user_slots)
        
        if len(common_slots) == 0:
            # print(f"User {i+2} has no matching times with the group.")
            return []

    # Now considering the preferences among common time slots
    ranked_slots = []
    # print(f"Common Slots: {common_slots} before preferences are chosen")
    # This multi-nested for loop basically just looks for preferences in the common slots among users. Most preferred slots get highest score.
    # Score is initialized to 0, and added 1 for each user who has a preference for that slot. The score is then used to sort the slots.
    for day, hour in common_slots:
        current_score = 0
        for user in users_data:
            found_preference = False
            for pref in user.preferred_slots:
                if pref.day == day and pref.hour == hour:
                    found_preference = True
                    break 
            if found_preference:
                current_score += 1
        
        ranked_slots.append({'day': day, 'hour': hour, 'score': current_score})

    
    day_mapping = {
        "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5
    }

    # We create a specific function just to handle the sorting logic.
    # Python will run this function on every single slot in the list 
    # to decide which one should go first.
    # The assumption is that, an earlier schedule in terms of day, hour, and score is better.
    def get_sorting_priorities(slot):
        
        # Python always sorts from SMALLEST number to LARGEST number (Ascending).
        # But we want the HIGHEST score first (Descending).
        # TRICK: We multiply by -1.
        #   Real Score: 5 (Best)  -> Sorting Value: -5  (Smallest number, comes first)
        #   Real Score: 0 (Worst) -> Sorting Value: 0   (Largest number, comes last)
        # print(f"Whole slot: {slot}")
        sort_by_score = slot['score'] * -1
        # print(f"Sort by score: {sort_by_score}")

        # If scores are tied, we look at this next.
        # Example: Monday (1) is smaller than Tuesday (2), so Monday comes first.
        sort_by_day = day_mapping[slot['day']]
        # print(f"Sort by day: {sort_by_day}")
        
        # If Day and Score are tied, we look at this last.
        # Example: 9am (9) is smaller than 10am (10), so 9am comes first.
        sort_by_hour = slot['hour']
        # print(f"Sort by hour: {sort_by_hour}")
        
        # We return a "tuple" of these three numbers.
        # Python compares the first number... if tied, compares the second... etc.
        return (sort_by_score, sort_by_day, sort_by_hour)

    # Run the sort using our custom logic function
    final_sorted_list = sorted(ranked_slots, key=get_sorting_priorities)
    # print(f"Final sorted list: {final_sorted_list}")
    
    return final_sorted_list

@app.route('/')
def home():
    return 'Scheduling API is running.'

@app.route('/schedule', methods=['POST'])
def schedule():
    try:
        data = request.get_json()
        if not data or 'messages' not in data:
            return jsonify({'error': 'Missing "messages" list in request body'}), 400
        
        msgs = data['messages']
        if not isinstance(msgs, list):
             return jsonify({'error': '"messages" must be a list of strings'}), 400

        users_data = []
        for i, m in enumerate(msgs):
            # print(f"Processing User {i+1}...")
            user_data = get_user_data(m)
            users_data.append(user_data)
        
        results = find_best_times(users_data)
        return jsonify({'recommended_times': results}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)