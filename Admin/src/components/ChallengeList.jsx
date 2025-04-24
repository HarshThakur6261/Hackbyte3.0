import { useEffect, useState } from 'react';
import { getChallenges, deleteChallenge } from '../services/challengeService';
import { 
  Table, TableBody, TableCell, TableContainer, 
  TableHead, TableRow, Paper, Button, 
  IconButton, Typography 
} from '@mui/material';
import { Delete, Edit } from '@mui/icons-material';

export default function ChallengeList({ onAddNew, onEdit }) {
  const [challenges, setChallenges] = useState([]);

  useEffect(() => {
    loadChallenges();
  }, []);

  const loadChallenges = async () => {
    const response = await getChallenges();
    setChallenges(response.data);
  };

  const handleDelete = async (id) => {
    await deleteChallenge(id);
    loadChallenges();
  };

  return (
    <div style={{ padding: '20px' }}>
      <Typography variant="h4" gutterBottom>
        Challenges
        <Button 
          onClick={onAddNew} 
          variant="contained" 
          color="primary"
          style={{ float: 'right' }}
        >
          Add New
        </Button>
      </Typography>
      
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Difficulty</TableCell>
              <TableCell>Min Stake</TableCell>
              <TableCell>Max Stake</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {challenges.map((challenge) => (
              <TableRow key={challenge._id}>
                <TableCell>{challenge.name}</TableCell>
                <TableCell>{challenge.type}</TableCell>
                <TableCell>{challenge.difficulty}</TableCell>
                <TableCell>${challenge.minStake}</TableCell>
                <TableCell>${challenge.maxStake}</TableCell>
                <TableCell>
                  <IconButton 
                    onClick={() => onEdit(challenge._id)}
                    color="primary"
                  >
                    <Edit />
                  </IconButton>
                  <IconButton 
                    onClick={() => handleDelete(challenge._id)}
                    color="error"
                  >
                    <Delete />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </div>
  );
}