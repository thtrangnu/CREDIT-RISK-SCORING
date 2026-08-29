import { BrowserRouter, Routes, Route } from 'react-router-dom';
import NavBar from './components/nav/NavBar';
import ScorePage from './pages/ScorePage';
import InsightsPage from './pages/InsightsPage';
import HistoryPage from './pages/HistoryPage';
import { ScoreProvider } from './context/ScoreContext';

export default function App() {
  return (
    <ScoreProvider>
      <BrowserRouter>
        <NavBar />
        <Routes>
          <Route path="/" element={<ScorePage />} />
          <Route path="/insights" element={<InsightsPage />} />
          <Route path="/history" element={<HistoryPage />} />
        </Routes>
      </BrowserRouter>
    </ScoreProvider>
  );
}
