import React, { Component } from 'react';
import { Route, Switch } from 'react-router-dom';
import Layout from './components/Navigation/Layout';
import { Home } from './components/Home';
import { FetchData } from './components/FetchData';
import { Counter } from './components/Counter';
import {
  HTTPCode400,
  HTTPCode401,
  HTTPCode403,
  HTTPCode404,
  HTTPCode500,
} from './components/AJAX/HTTPCodes/HTTPCodes.js';
import './custom.css'
// to enable cookies, uncomment below
//import { withCookies } from 'react-cookie'; // Cookies logic implented

export default class App extends Component {
  static displayName = App.name;

  render () {
    return (
      <div className="App">

        {/* Excellent answer on rendering a / route */}
        {/* https://stackoverflow.com/a/44292410/9505707 */}
        <Layout />
        <Switch>
          <Route exact path='/' component={Home} />
          <Route path='/counter' component={Counter} />
          <Route path='/fetch-data' component={FetchData} />
          {/*  Error routes */}
          <Route path='/error400' component={HTTPCode400} />
          <Route path='/error401' component={HTTPCode401} />
          <Route path='/error403' component={HTTPCode403} />
          <Route path='/error404' component={HTTPCode404} />
          <Route path='/error500' component={HTTPCode500} />
          {/* Default route on no other route found. */}
          <Route component={HTTPCode400} />
        </Switch>
      </div>
    );
  }
}


// see this link below on setting up cookies with React
// https://medium.com/@rossbulat/using-cookies-in-react-redux-and-react-router-4-f5f6079905dc
// export default withCookies(App);
