import axios from "axios";

const API_URL = "http://18.234.231.208/api/v1";

export const analyzeSentiment = async (text) => {
  try {
    const res = await axios.post(`${API_URL}/predict`, { text });
    return res.data;
  } catch (err) {
    console.error(err);
    return null;
  }
};
