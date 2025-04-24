import { useState } from 'react';
import ChallengeList from './components/ChallengeList';
import ChallengeForm from './components/ChallengeForm';

function App() {
  const [currentView, setCurrentView] = useState('list'); // 'list' or 'form'
  const [editingChallengeId, setEditingChallengeId] = useState(null);

  // Handle view transitions
  const showChallengeList = () => {
    setCurrentView('list');
    setEditingChallengeId(null);
  };

  const showNewChallengeForm = () => {
    setCurrentView('form');
    setEditingChallengeId(null);
  };

  const showEditChallengeForm = (id) => {
    setCurrentView('form');
    setEditingChallengeId(id);
  };

  return (
    <div className="app-container">
      {currentView === 'list' ? (
        <ChallengeList
          onAddNew={showNewChallengeForm}
          onEdit={showEditChallengeForm}
        />
      ) : (
        <ChallengeForm
          id={editingChallengeId}
          onSave={showChallengeList}
          onCancel={showChallengeList}
        />
      )}
    </div>
  );
}

export default App;