import { createContext, useContext, useState } from 'react';

const ScoreContext = createContext(null);

export function ScoreProvider({ children }) {
  const [applicant, setApplicant] = useState(null);
  const [scoreResult, setScoreResult] = useState(null);

  return (
    <ScoreContext.Provider value={{ applicant, setApplicant, scoreResult, setScoreResult }}>
      {children}
    </ScoreContext.Provider>
  );
}

export function useScoreContext() {
  const ctx = useContext(ScoreContext);
  if (!ctx) throw new Error('useScoreContext phải dùng trong <ScoreProvider>');
  return ctx;
}
