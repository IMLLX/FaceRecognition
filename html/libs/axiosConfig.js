axios.defaults.baseURL = 'http://127.0.0.1:8080';
axios.defaults.responseType = "json";
axios.defaults.headers.post = {
    'Content-Type': 'multipart/form-data'
}
axios.defaults.transformRequest = (data) => {
    return data
}