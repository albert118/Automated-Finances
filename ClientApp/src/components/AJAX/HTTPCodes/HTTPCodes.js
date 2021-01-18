import React, { Component } from 'react';
import { Button } from 'react-bootstrap';
import { Container } from 'reactstrap';

function logErrorMessageToConsole(message) {
	// TODO print stacktrace
	var printStatement = message ? String(message) : "DEBUGGER";
	console.log(printStatement);
}

const errorMessageString = (errorMessage, errorCode) => {
	return (errorMessage ? errorCode + ' ' + errorMessage : "Unkown Error. Please consult the stack trace.");
}

export default function BaseError(props) {
	const titleErrorCode = props.errorCode ? props.errorCode : "Unknown Error Code";
	// TODO: Replace this with component
	const reportBugMsg = "Report Bug";
	const returnHomeMsg = "Return Home";
	const _errMsg = errorMessageString(props.errorMessage, props.errorCode);

	logErrorMessageToConsole(_errMsg);

	return (
		<Container>
			<div className="http-error-frame">
				<header>
					<h1 className="error-code-title"> { titleErrorCode } </h1>
				</header>
				<div className="errorMessage-frame">
					<p className="subtext"> { _errMsg } </p>
				</div>

				//TODO: Replace this with component

				<div className="contactUs-frame">
					<button href="/" className="btn btn-primary">
						{returnHomeMsg}
					</button>	
					<br></br>
					<button href="https://github.com/albert118/Automated-Finances/issues" className="btn btn-primary">
						{ reportBugMsg }
					</button>
				</div>
			</div>
		</Container>
	);
}

export function HTTPCode400() {
	const errorCode = 400;
	const errorMessage = "BAD REQUEST";

	return (
		<BaseError errorCode={errorCode} errorMessage={errorMessage} />
	);
}

export function HTTPCode401() {
	const errorCode = 401;
	const errorMessage = "UNAUTHORISED";

	return (
		<BaseError errorCode={errorCode} errorMessage={errorMessage} />
	);
}

export function HTTPCode403() {
	const errorCode = 403;
	const errorMessage = "FORBIDDEN";

	return (
		<BaseError errorCode={errorCode} errorMessage={errorMessage} />
	);
}

export function HTTPCode404() {
	const errorCode = 404;
	const errorMessage = "NOT FOUND";

	return (
		<BaseError errorCode={errorCode} errorMessage={errorMessage} />
	);
}

export function HTTPCode500() {
	const errorCode = 500;
	const errorMessage = "INTERNAL SERVER ERROR";

	return (
		<BaseError errorCode={errorCode} errorMessage={errorMessage} />
	);
}
