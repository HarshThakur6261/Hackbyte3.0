// import { useState, useEffect } from 'react';
// import { useNavigate, useParams } from 'react-router-dom';
// import { 
//   TextField, Button, Typography, Paper, 
//   FormControl, InputLabel, Select, MenuItem, 
//   Chip, Box, Grid 
// } from '@mui/material';
// import { 
//   getChallenge, createChallenge, updateChallenge 
// } from '../services/challengeService';

// const difficultyOptions = ['easy', 'medium', 'hard'];
// const typeOptions = ['steps', 'strength', 'combo'];
// const tagOptions = ['cardio', 'strength', 'beginner', 'advanced', 'daily', 'weekly'];

// export default function ChallengeForm() {
//   const { id } = useParams();
//   const navigate = useNavigate();
//   const [challenge, setChallenge] = useState({
//     name: '',
//     description: '',
//     type: '',
//     stepGoal: 0,
//     exercises: [{ name: '', reps: 0, videoDemoURL: '' }],
//     minStake: 0,
//     maxStake: 50,
//     rewardMultiplier: 1.0,
//     difficulty: '',
//     tags: []
//   });

//   useEffect(() => {
//     if (id) {
//       loadChallenge(id);
//     }
//   }, [id]);

//   const loadChallenge = async (id) => {
//     const response = await getChallenge(id);
//     setChallenge(response.data);
//   };

//   const handleChange = (e) => {
//     setChallenge({ ...challenge, [e.target.name]: e.target.value });
//   };

//   const handleExerciseChange = (index, e) => {
//     const newExercises = [...challenge.exercises];
//     newExercises[index] = { ...newExercises[index], [e.target.name]: e.target.value };
//     setChallenge({ ...challenge, exercises: newExercises });
//   };

//   const handleAddExercise = () => {
//     setChallenge({ 
//       ...challenge, 
//       exercises: [...challenge.exercises, { name: '', reps: 0}] 
//     });
//   };

//   const handleRemoveExercise = (index) => {
//     const newExercises = challenge.exercises.filter((_, i) => i !== index);
//     setChallenge({ ...challenge, exercises: newExercises });
//   };

//   const handleTagChange = (event) => {
//     setChallenge({ ...challenge, tags: event.target.value });
//   };

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     try {
//       if (id) {
//         await updateChallenge(id, challenge);
//       } else {
//         await createChallenge(challenge);
//       }
//       navigate('/challenges');
//     } catch (error) {
//       console.error('Error saving challenge:', error);
//     }
//   };

//   return (
//     <Paper style={{ padding: '20px', margin: '20px 0' }}>
//       <Typography variant="h5" gutterBottom>
//         {id ? 'Edit Challenge' : 'Create New Challenge'}
//       </Typography>

//       <form onSubmit={handleSubmit}>
//         <Grid container spacing={3}>
//           <Grid item xs={12} md={6}>
//             <TextField
//               fullWidth
//               label="Name"
//               name="name"
//               value={challenge.name}
//               onChange={handleChange}
//               required
//             />
//           </Grid>

//           <Grid item xs={12} md={6}>
//             <FormControl fullWidth>
//               <InputLabel>Type</InputLabel>
//               <Select
//                 name="type"
//                 value={challenge.type}
//                 onChange={(e) => setChallenge({ ...challenge, type: e.target.value })}
//                 required
//               >
//                 {typeOptions.map((type) => (
//                   <MenuItem key={type} value={type}>{type}</MenuItem>
//                 ))}
//               </Select>
//             </FormControl>
//           </Grid>

//           <Grid item xs={12} md={6}>
//             <TextField
//               fullWidth
//               label="Step Goal"
//               name="stepGoal"
//               type="number"
//               value={challenge.stepGoal}
//               onChange={handleChange}
//               disabled={challenge.type === 'strength'}
//             />
//           </Grid>

//           <Grid item xs={12} md={6}>
//             <FormControl fullWidth>
//               <InputLabel>Difficulty</InputLabel>
//               <Select
//                 name="difficulty"
//                 value={challenge.difficulty}
//                 onChange={(e) => setChallenge({ ...challenge, difficulty: e.target.value })}
//                 required
//               >
//                 {difficultyOptions.map((level) => (
//                   <MenuItem key={level} value={level}>{level}</MenuItem>
//                 ))}
//               </Select>
//             </FormControl>
//           </Grid>

//           <Grid item xs={12} md={6}>
//             <TextField
//               fullWidth
//               label="Minimum Stake ($)"
//               name="minStake"
//               type="number"
//               value={challenge.minStake}
//               onChange={handleChange}
//               inputProps={{
//                 min: "0.000001",
//                 step: "0.000001",  // Must match the precision you want
//                 inputMode: "decimal",
//                 pattern: "^\\d*\\.?\\d{0,6}$" // Allows up to 6 decimals
//               }}
//               required
//             />
//           </Grid>
//           <Grid item xs={12} md={6}>
//             <TextField
//               fullWidth
//               label="Maximum Stake ($)"
//               name="maxStake"
//               type="number"
//               value={challenge.maxStake}
//               onChange={handleChange}
//               inputProps={{
//                 max: "50",
//                 step: "0.000001",  // Must match the precision you want
//                 inputMode: "decimal",
//                 pattern: "^\\d*\\.?\\d{0,6}$" // Allows up to 6 decimals
//               }}
//               required
//             />
//           </Grid>

//           <Grid item xs={12} md={6}>
//             <TextField
//               fullWidth
//               label="Reward Multiplier"
//               name="rewardMultiplier"
//               type="number"
//               value={challenge.rewardMultiplier}
//               onChange={handleChange}
//               inputProps={{ min: 1.0, step: 0.1 }}
//               required
//             />
//           </Grid>

//           <Grid item xs={12}>
//             <FormControl fullWidth>
//               <InputLabel>Tags</InputLabel>
//               <Select
//                 multiple
//                 value={challenge.tags}
//                 onChange={handleTagChange}
//                 renderValue={(selected) => (
//                   <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
//                     {selected.map((value) => (
//                       <Chip key={value} label={value} />
//                     ))}
//                   </Box>
//                 )}
//               >
//                 {tagOptions.map((tag) => (
//                   <MenuItem key={tag} value={tag}>{tag}</MenuItem>
//                 ))}
//               </Select>
//             </FormControl>
//           </Grid>

//           <Grid item xs={12}>
//             <Typography variant="h6" gutterBottom>
//               Exercises
//             </Typography>
//             {challenge.exercises.map((exercise, index) => (
//               <Box key={index} mb={2} p={2} border={1} borderRadius={2}>
//                 <Grid container spacing={2}>
//                   <Grid item xs={12} md={4}>
//                     <TextField
//                       fullWidth
//                       label="Exercise Name"
//                       name="name"
//                       value={exercise.name}
//                       onChange={(e) => handleExerciseChange(index, e)}
//                       required
//                     />
//                   </Grid>
//                   <Grid item xs={12} md={4}>
//                     <TextField
//                       fullWidth
//                       label="Reps"
//                       name="reps"
//                       type="number"
//                       value={exercise.reps}
//                       onChange={(e) => handleExerciseChange(index, e)}
//                       inputProps={{ min: 0 }}
//                       required
//                     />
//                   </Grid>

//                   <Grid item xs={12} md={1}>
//                     <Button 
//                       onClick={() => handleRemoveExercise(index)}
//                       color="error"
//                       disabled={challenge.exercises.length <= 1}
//                     >
//                       Remove
//                     </Button>
//                   </Grid>
//                 </Grid>
//               </Box>
//             ))}
//             <Button onClick={handleAddExercise} variant="outlined">
//               Add Exercise
//             </Button>
//           </Grid>

//           <Grid item xs={12}>
//             <Button type="submit" variant="contained" color="primary">
//               Save Challenge
//             </Button>
//           </Grid>
//         </Grid>
//       </form>
//     </Paper>
//   );
// }

import { useState, useEffect } from 'react';
import {
    TextField, Button, Typography, Paper,
    FormControl, InputLabel, Select, MenuItem,
    Chip, Box, Grid
} from '@mui/material';
import {
    getChallenge, createChallenge, updateChallenge
} from '../services/challengeService';

const difficultyOptions = ['easy', 'medium', 'hard'];
const typeOptions = ['steps', 'strength', 'combo'];
const tagOptions = ['cardio', 'strength', 'beginner', 'advanced', 'daily', 'weekly'];

export default function ChallengeForm({ id, onSave }) {
    const [challenge, setChallenge] = useState({
        name: '',
        description: '',
        type: '',
        stepGoal: 0,
        exercises: [{ name: '', reps: 0, videoDemoURL: '' }],
        minStake: 0,
        maxStake: 50,
        rewardMultiplier: 1.0,
        difficulty: '',
        tags: []
    });

    useEffect(() => {
        if (id) {
            loadChallenge(id);
        }
    }, [id]);

    const loadChallenge = async (id) => {
        const response = await getChallenge(id);
        setChallenge(response.data);
    };

    const handleChange = (e) => {
        setChallenge({ ...challenge, [e.target.name]: e.target.value });
    };

    const handleExerciseChange = (index, e) => {
        const newExercises = [...challenge.exercises];
        newExercises[index] = { ...newExercises[index], [e.target.name]: e.target.value };
        setChallenge({ ...challenge, exercises: newExercises });
    };

    const handleAddExercise = () => {
        setChallenge({
            ...challenge,
            exercises: [...challenge.exercises, { name: '', reps: 0 }]
        });
    };

    const handleRemoveExercise = (index) => {
        const newExercises = challenge.exercises.filter((_, i) => i !== index);
        setChallenge({ ...challenge, exercises: newExercises });
    };

    const handleTagChange = (event) => {
        setChallenge({ ...challenge, tags: event.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            if (id) {
                await updateChallenge(id, challenge);
            } else {
                await createChallenge(challenge);
            }
            onSave(); // Call the parent's save handler
        } catch (error) {
            console.error('Error saving challenge:', error);
        }
    };

    return (
        <Paper style={{ padding: '20px', margin: '20px 0' }}>
            <Typography variant="h5" gutterBottom>
                {id ? 'Edit Challenge' : 'Create New Challenge'}
            </Typography>

            <form onSubmit={handleSubmit}>
                <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                        <TextField
                            fullWidth
                            label="Name"
                            name="name"
                            value={challenge.name}
                            onChange={handleChange}
                            required
                        />
                    </Grid>

                    <Grid item xs={12} md={6}>
                        <FormControl fullWidth>
                            <InputLabel>Type</InputLabel>
                            <Select
                                name="type"
                                value={challenge.type}
                                onChange={(e) => setChallenge({ ...challenge, type: e.target.value })}
                                required
                            >
                                {typeOptions.map((type) => (
                                    <MenuItem key={type} value={type}>{type}</MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Grid>

                    <Grid item xs={12} md={6}>
                        <TextField
                            fullWidth
                            label="Step Goal"
                            name="stepGoal"
                            type="number"
                            value={challenge.stepGoal}
                            onChange={handleChange}
                            disabled={challenge.type === 'strength'}
                        />
                    </Grid>

                    <Grid item xs={12} md={6}>
                        <FormControl fullWidth>
                            <InputLabel>Difficulty</InputLabel>
                            <Select
                                name="difficulty"
                                value={challenge.difficulty}
                                onChange={(e) => setChallenge({ ...challenge, difficulty: e.target.value })}
                                required
                            >
                                {difficultyOptions.map((level) => (
                                    <MenuItem key={level} value={level}>{level}</MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Grid>

                    <Grid item xs={12} md={6}>
                        <TextField
                            fullWidth
                            label="Minimum Stake ($)"
                            name="minStake"
                            type="number"
                            value={challenge.minStake}
                            onChange={handleChange}
                            inputProps={{
                                min: "0.000001",
                                step: "0.000001",  // Must match the precision you want
                                inputMode: "decimal",
                                pattern: "^\\d*\\.?\\d{0,6}$" // Allows up to 6 decimals
                            }}
                            required
                        />
                    </Grid>
                    <Grid item xs={12} md={6}>
                        <TextField
                            fullWidth
                            label="Maximum Stake ($)"
                            name="maxStake"
                            type="number"
                            value={challenge.maxStake}
                            onChange={handleChange}
                            inputProps={{
                                max: "50",
                                step: "0.000001",  // Must match the precision you want
                                inputMode: "decimal",
                                pattern: "^\\d*\\.?\\d{0,6}$" // Allows up to 6 decimals
                            }}
                            required
                        />
                    </Grid>

                    <Grid item xs={12} md={6}>
                        <TextField
                            fullWidth
                            label="Reward Multiplier"
                            name="rewardMultiplier"
                            type="number"
                            value={challenge.rewardMultiplier}
                            onChange={handleChange}
                            inputProps={{ min: 1.0, step: 0.1 }}
                            required
                        />
                    </Grid>

                    <Grid item xs={12}>
                        <FormControl fullWidth>
                            <InputLabel>Tags</InputLabel>
                            <Select
                                multiple
                                value={challenge.tags}
                                onChange={handleTagChange}
                                renderValue={(selected) => (
                                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                                        {selected.map((value) => (
                                            <Chip key={value} label={value} />
                                        ))}
                                    </Box>
                                )}
                            >
                                {tagOptions.map((tag) => (
                                    <MenuItem key={tag} value={tag}>{tag}</MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Grid>

                    <Grid item xs={12}>
                        <Typography variant="h6" gutterBottom>
                            Exercises
                        </Typography>
                        {challenge.exercises.map((exercise, index) => (
                            <Box key={index} mb={2} p={2} border={1} borderRadius={2}>
                                <Grid container spacing={2}>
                                    <Grid item xs={12} md={4}>
                                        <TextField
                                            fullWidth
                                            label="Exercise Name"
                                            name="name"
                                            value={exercise.name}
                                            onChange={(e) => handleExerciseChange(index, e)}
                                            required
                                        />
                                    </Grid>
                                    <Grid item xs={12} md={4}>
                                        <TextField
                                            fullWidth
                                            label="Reps"
                                            name="reps"
                                            type="number"
                                            value={exercise.reps}
                                            onChange={(e) => handleExerciseChange(index, e)}
                                            inputProps={{ min: 0 }}
                                            required
                                        />
                                    </Grid>

                                    <Grid item xs={12} md={1}>
                                        <Button
                                            onClick={() => handleRemoveExercise(index)}
                                            color="error"
                                            disabled={challenge.exercises.length <= 1}
                                        >
                                            Remove
                                        </Button>
                                    </Grid>
                                </Grid>
                            </Box>
                        ))}
                        <Button onClick={handleAddExercise} variant="outlined">
                            Add Exercise
                        </Button>
                    </Grid>
                    <Grid item xs={12}>
                        <Button type="submit" variant="contained" color="primary">
                            Save Challenge
                        </Button>
                    </Grid>
                </Grid>
            </form>
        </Paper>
    );
}