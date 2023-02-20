#### Running the WebApp Locally

Download this repository and ensure you have the dependencies listed in [requirements.txt](requirements.txt).

To run the Streamlit app, run:<br>
`streamlit run app.py`


To run as a Docker app, run:<br>

`sudo docker build -t capstone_streamlit .` <br>
`sudo docker run -p 8501:8501 capstone_streamlit` <br>
or <br>
`docker build -t capstone_streamlit .` <br>
`docker run -p 8501:8501 capstone_streamlit` <br>

