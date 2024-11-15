**EEG Alzheimer's Detection Using LLM and ML Classifier**

This project is designed to analyze EEG data and predict the likelihood of Alzheimer's disease using a combination of a Large Language Model (LLM) and a Machine Learning (ML) classifier. The tool provides an interface for uploading EEG data and receiving diagnostic predictions.

**Features**

**EEG Data Upload:** Easily upload EEG files for analysis.

**Integrated Analysis:** Utilizes both LLM and ML models to enhance prediction accuracy.

**User Interface:** Built with the "gradio" package for a seamless user experience.

**Integration with OpenAI and NVIDIA:** This project utilizes NVIDIA's Palmyra-Med-70B-32K API in collaboration with OpenAI's advanced language models to enhance the analysis and prediction capabilities of the system.

**Installation**
Clone the Repository:

bash

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Install Dependencies:

Ensure you have Python installed.
Install the necessary packages using the provided text file:
bash

pip install -r requirements.txt
Set Up API Key:

Obtain an API key from [nVIdia](https://build.nvidia.com/explore/discover) by making an acount.
Create a .env file in the project folder and add your API key:
SECRET_API_KEY=your_api_key_here
Usage
Run the Main Function:

Execute the main script to start the application:
bash

python main.py
Interface Interaction:

Use the user interface to upload EEG data and receive predictions.
Contributing
We welcome contributions from the community. Please feel free to submit issues or pull requests.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For questions or support, please contact Soudeh (Venus) Mostaghimi at mostaghs@uci.edu, or Trisha Mendoza at trisham2uci.edu, or Blanca Romer at blancr1@uci.edu.
