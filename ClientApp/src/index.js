import 'bootstrap/dist/css/bootstrap.css';
import React from 'react';
import { render } from 'react-dom';
import { BrowserRouter as Router } from 'react-router-dom';
import App from './App';
import registerServiceWorker from './registerServiceWorker';
//import { CookiesProvider } from 'react-cookie'; // Cookies logic implented

const baseUrl = document.getElementsByTagName('base')[0].getAttribute('href');

render(
  <Router basename={baseUrl}>
    <App />
  </Router>,
  document.getElementById('root')
);

registerServiceWorker();

