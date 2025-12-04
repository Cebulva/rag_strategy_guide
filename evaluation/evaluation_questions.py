EVALUATION_DATA: list[dict[str, str]] = [
    {
        "question": "What is inside of the red book?",
        "ground_truth": "The red book has a message. A young man struggling to see and "
                        "hear me. He began demanding the red pages—thought he said something "
                        "like I am Sirrus. The son of whom Atrus spoke? "
    },
    {
        "question": "What is inside of The Dock Forechamber?",
        "ground_truth": """
                        The Dock Forechamber contains a device called the Dimensional Imager.

                        You can interact with the Dimensional Imager in several ways:
                        1.  Open the front cover: The control panel for the Imager is located on the wall by the exit. You need to click on the button at the upper left of this control panel to open the front cover of the Imager.
                        2.  Enter specific codes: Once the cover is open, you can enter three specific codes listed on the cover of the panel: 40, 47, and 67. After entering a code, you can view the corresponding images by pressing the button on the front of the Imager itself.
                        3.  View an additional 3-D image: There's a special image you can view by entering the number of Marker Switches found on the island. While you could count them yourself, the guide tells us there are **8** Marker Switches. So, you can enter "08" into the Imager's control panel to view this additional 3-D image, which allows you to "meet Atrus."
                        """
    },
]