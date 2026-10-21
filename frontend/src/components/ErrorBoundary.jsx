import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error('Page error:', error, info);
  }

  handleRetry = () => {
    this.setState({ error: null });
    if (this.props.onRetry) this.props.onRetry();
  };

  render() {
    if (this.state.error) {
      return (
        <div className="page-error glass-panel" role="alert">
          <AlertTriangle size={22} aria-hidden />
          <div>
            <h2 className="page-title">This page could not be displayed</h2>
            <p className="page-subtitle">{this.state.error?.message || 'An unexpected error occurred.'}</p>
            <button type="button" className="btn btn-primary" onClick={this.handleRetry} style={{ marginTop: 12 }}>
              <RefreshCw size={14} aria-hidden /> Try again
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
