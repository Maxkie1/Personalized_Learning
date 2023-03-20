"""This module contains moodle API functionalities."""

import requests
from bs4 import BeautifulSoup
import dotenv
import os
from pprint import pprint

dotenv.load_dotenv()
PERSONALIZED_LEARNING_TOKEN = os.getenv("PERSONALIZED_LEARNING_TOKEN")
MOODLE_USERNAME = os.getenv("MOODLE_USERNAME")
MOODLE_PASSWORD = os.getenv("MOODLE_PASSWORD")


def get_enrolled_users(course_id):
    """Get enrolled users of a course.

    Args:
        course_id: The course ID.

    Returns:
        A list of enrolled users.
    """

    url = f"http://ai-in-education.dhbw-stuttgart.de/moodle/webservice/rest/server.php"
    params = {
        "wstoken": PERSONALIZED_LEARNING_TOKEN,
        "wsfunction": "core_enrol_get_enrolled_users",
        "courseid": course_id,
        "moodlewsrestformat": "json",
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        raise Exception("Error while retrieving enrolled users.")


def get_user_activities_completion(user_id, course_id):
    """Get user activities completion status within a course for a specific user.

    Args:
        user_id: The ID of the user whose activities completion status should be retrieved.
        course_id: The ID of the course for which the activities completion status should be retrieved.

    Returns:
        A list of activities completion status.
    """

    url = f"http://ai-in-education.dhbw-stuttgart.de/moodle/webservice/rest/server.php"
    params = {
        "wstoken": PERSONALIZED_LEARNING_TOKEN,
        "wsfunction": "core_completion_get_activities_completion_status",
        "courseid": course_id,
        "userid": user_id,
        "moodlewsrestformat": "json",
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        raise Exception("Error while retrieving user activities completion status.")


def session_login():
    """Login to Moodle to enable further requests.

    Returns:
        The Moodle session used for further requests.
    """

    # login credentials
    login_url = f"http://ai-in-education.dhbw-stuttgart.de/moodle/login/index.php"
    login_data = {
        "username": MOODLE_USERNAME,
        "password": MOODLE_PASSWORD,
    }

    # send a GET request to the login page to extract the login token
    session = requests.Session()
    response = session.get(login_url)
    soup = BeautifulSoup(response.content, "html.parser")
    login_token = soup.find("input", {"name": "logintoken"})["value"]

    # add the login token to the login data dictionary
    login_data["logintoken"] = login_token

    # send a POST request to the login page with the login credentials
    response = session.post(login_url, data=login_data)

    # check if the login was successful (status code 200)
    if response.status_code == 200:
        return session
    else:
        raise Exception("Failed to login")


def session_logout(session):
    """Logout from Moodle to end the session.

    Args:
        session: The Moodle session to be logged out.
    """

    search_url = f"http://ai-in-education.dhbw-stuttgart.de/moodle/my/"
    search = session.get(search_url)
    soup = BeautifulSoup(search.content, "html.parser")
    logout_url = soup.find("a", {"data-title": "logout,moodle"})["href"]

    response = session.get(logout_url)

    if response.status_code == 200:
        session.close()
    else:
        raise Exception("Failed to logout")


def moodle_access_user_logs(user_id, course_id, session):
    """Access the user logs pages of a course for a specific user.

    The user logs pages contains the activity logs of a user within a course.
    The pages are parsed with BeautifulSoup to extract the activity logs.

    Args:
        user_id: The ID of the user whose user logs pages should be accessed.
        course_id: The ID of the course for which the user logs pages should be accessed.
        session: The already logged in Moodle session.

    Returns:
        A list oof user logs pages as BeautifulSoup objects.
    """

    # create a GET request to the user logs page with the session
    url = f"http://ai-in-education.dhbw-stuttgart.de/moodle/report/log/user.php"
    params = {
        "id": user_id,
        "course": course_id,
        "mode": "all",
        "logreader": "logstore_standard",
        "page": 0,
    }
    response = session.get(url, params=params)

    # check if the response was successful (status code 200)
    if response.status_code == 200:
        # parse the response with BeautifulSoup
        first_visit = BeautifulSoup(response.content, "html.parser")

        # find the number of pages
        pagination = first_visit.find("nav", {"class": "pagination"})
        if pagination:
            num_pages = int(pagination.find_all("li")[-2].text)
        else:
            num_pages = 1

        # extract the soup objects of all pages
        soups = []
        for page in range(num_pages):
            params["page"] = page
            response = session.get(url, params=params)
            soups.append(BeautifulSoup(response.content, "html.parser"))

        return soups
    else:
        raise Exception("Failed to retrieve user logs")


def extract_user_activity_logs(soups):
    """Extract the user activity logs from the user logs pages.

    Args:
        soups: A list of parsed user logs pages as BeautifulSoup objects.

    Returns:
        A list of user activity logs.
    """

    activity_logs = []

    for soup in soups:
        # find the table with the activity logs
        table = soup.find("table", {"class": "generaltable"})

        # find all table rows
        rows = table.find_all("tr")

        # extract the activity logs
        for row in rows[1:]:  # skip the first row (table header)
            # find all table cells
            cells = row.find_all("td")
            # check if the event is relevant
            if cells[5].text == "Course module viewed":
                activity = create_activity_log(cells)
                activity_logs.append(activity)

    return activity_logs


def create_activity_log(cells):
    """Create an activity log from a table row of the user logs page.

    Args:
        cells: The table cells of the table row.

    Returns:
        A dictionary containing the activity data.
    """

    activity = {
        "time": cells[0].text,
        "user full name": cells[1].text,
        "event context": cells[3].text,
        "component": cells[4].text,
        "event name": cells[5].text,
        "description": cells[6].text,
    }

    if (
        "URL" in cells[3].text or "File" in cells[3].text or "Folder" in cells[3].text
    ) and "Video" in cells[3].text:
        # add new value to component key
        activity["component"] = "Video"
    elif (
        "URL" in cells[3].text or "File" in cells[3].text or "Folder" in cells[3].text
    ) and "Image" in cells[3].text:
        activity["component"] = "Image"
    elif "Forum" in cells[3].text and "FAQ" in cells[3].text:
        activity["component"] = "FAQ"
    elif "Page" in cells[3].text and "Example" in cells[3].text:
        activity["component"] = "Example"

    return activity


def get_user_activity_logs(user_id, course_id):
    """Get user activity logs within a course for a specific user.

    Args:
        user_id: The ID of the user whose activity logs are to be retrieved.
        course_id: The ID of the course for which the activity logs are to be retrieved.

    Returns:
        A list of user activity logs.
    """

    # login to Moodle
    session = session_login()

    # access the user logs pages of a course
    soups = moodle_access_user_logs(user_id, course_id, session)

    # extract the user activity logs from the user logs pages
    activity_logs = extract_user_activity_logs(soups)

    # logout from Moodle
    session_logout(session)

    return activity_logs


def aggregate_user_activity_logs(course_id):
    """Aggregate the user activity logs of all users within a course.

    Args:
        course_id: The ID of the course for which the user activity logs are to be aggregated.

    Returns:
        A list of user activity logs of all enrolled users within a course.
    """

    # get all enrolled users within a course
    enrolled_users = get_enrolled_users(course_id)

    # get the user activity logs of all enrolled users within a course
    activity_logs = []
    for user in enrolled_users:
        activity_logs.extend(get_user_activity_logs(user["id"], course_id))

    return activity_logs


def add_user_to_group(user_id, learning_style):
    """Add a user to a learning style group.

    Each group represents a learning style. The user is added to respective group of his/her learning style.
    The student has access to learning style related activies and resources based on the group membership.
    The following groups are available:
        - 1: sensing
        - 2: intuitive
        - 3: visual
        - 4: verbal
        - 5: active
        - 6: reflective
        - 7: sequential
        - 8: global

    Args:
        user_id: The ID of the user to be added to a group.
        learning_style: The learning style of the user.
    """

    url = f"http://ai-in-education.dhbw-stuttgart.de/moodle/webservice/rest/server.php"
    params = {
        "wstoken": PERSONALIZED_LEARNING_TOKEN,
        "wsfunction": "core_group_add_group_members",
        "members[0][userid]": user_id,
        "members[0][groupid]": learning_style,
        "moodlewsrestformat": "json",
    }

    response = requests.post(url, params=params)

    if response.status_code != 200:
        raise Exception("Failed to add user to group")


def delete_user_from_group(user_id, learning_style):
    """Delete a user from a learning style group.

    Args:
        user_id: The ID of the user to be deleted from a group.
        learning_style: The learning style of the user.
    """

    url = f"http://ai-in-education.dhbw-stuttgart.de/moodle/webservice/rest/server.php"
    params = {
        "wstoken": PERSONALIZED_LEARNING_TOKEN,
        "wsfunction": "core_group_delete_group_members",
        "members[0][userid]": user_id,
        "members[0][groupid]": learning_style,
        "moodlewsrestformat": "json",
    }

    response = requests.post(url, params=params)

    if response.status_code != 200:
        raise Exception("Failed to delete user from group")


def get_group_members(group_ids):
    """Get the members of all learning style groups.

    Args:
        group_ids: The IDs of the groups whose members are to be retrieved.

    Returns:
        A list of all group members sorted by group ID.
    """

    url = f"http://ai-in-education.dhbw-stuttgart.de/moodle/webservice/rest/server.php"
    params = {
        "wstoken": PERSONALIZED_LEARNING_TOKEN,
        "wsfunction": "core_group_get_group_members",
        "groupids[0]": group_ids[0],
        "groupids[1]": group_ids[1],
        "groupids[2]": group_ids[2],
        "groupids[3]": group_ids[3],
        "groupids[4]": group_ids[4],
        "groupids[5]": group_ids[5],
        "groupids[6]": group_ids[6],
        "groupids[7]": group_ids[7],
        "moodlewsrestformat": "json",
    }

    response = requests.post(url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        raise Exception("Failed to get group members")


def is_user_in_group(user_id):
    """Check if a user is in any group.

    Args:
        user_id: The ID of the user to be checked.

    Returns:
        True if the user is in any group, False otherwise. If True, the respective group ID is also returned.
    """

    group_ids = [1, 2, 3, 4, 5, 6, 7, 8]

    group_members = get_group_members(group_ids)
    for group in group_members:
        if user_id in group["userids"]:
            return True, group["groupid"]

    return False, None


def learning_style_assignment(predicted_ls, user_id):
    """Assign a user to a new learning style group.

    Args:
        predicted_ls: The predicted learning style of the user.
        user_id: The ID of the user whose learning style is predicted.
    """

    is_group_member, previous_ls = is_user_in_group(user_id)

    if is_group_member and previous_ls == predicted_ls:
        print(
            "The user's previous learning style {} is the same as the new learning style {}.".format(
                previous_ls, predicted_ls
            )
        )
        return
    elif is_group_member:
        print(
            "The user's previous learning style {} is overwritten by learning style {}.".format(
                previous_ls, predicted_ls
            )
        )
        delete_user_from_group(user_id, previous_ls)
    else:
        print(
            "First learning style assignment for the user with learning style {}.".format(
                predicted_ls
            )
        )

    add_user_to_group(user_id, predicted_ls)