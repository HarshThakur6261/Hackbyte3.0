import { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';

const useFitnessData = () => {
    const [todaySteps, setTodaySteps] = useState(0);
    const [weeklySteps, setWeeklySteps] = useState(0);
    const [todayCalories, setTodayCalories] = useState(0);
    const [weeklyCalories, setWeeklyCalories] = useState(0);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const { token } = useAuth();

    useEffect(() => {
        if (!token) return;

        let isMounted = true;
        let intervalId = null;

        const fetchTodaySteps = async () => {
            try {
                const now = new Date();
                const todayStart = new Date(now);
                todayStart.setHours(0, 0, 0, 0);

                const todayEnd = now.getTime();

                const today = await fetchStepCount(token, todayStart.getTime(), todayEnd);
                if (isMounted) {
                    setTodaySteps(today);
                }
            } catch (error) {
                if (isMounted) {
                    console.error('Google Fit Error:', error);
                    setError(error.message || 'Failed to load today\'s fitness data');
                }
            }
        };

        const fetchInitialData = async () => {
            setIsLoading(true);
            try {
                const now = new Date();
                const todayStart = new Date(now);
                todayStart.setHours(0, 0, 0, 0);
                const todayEnd = now.getTime();

                const weekStart = new Date(todayStart);
                weekStart.setDate(weekStart.getDate() - 7);

                const [steps, calories] = await Promise.all([
                    fetchSteps(token, todayStart.getTime(), todayEnd, weekStart.getTime()),
                    fetchCalories(token, todayStart.getTime(), todayEnd, weekStart.getTime())
                ]);

                if (isMounted) {
                    setTodaySteps(steps.today);
                    setWeeklySteps(steps.weekly);
                    setTodayCalories(calories.today);
                    setWeeklyCalories(calories.weekly);
                }
            } catch (error) {
                if (isMounted) {
                    console.error('Google Fit Error:', error);
                    setError(error.message || 'Failed to load fitness data');
                }
            } finally {
                if (isMounted) setIsLoading(false);
            }
        };

        fetchInitialData();

        // Start interval to update today's steps every 30 seconds
        intervalId = setInterval(fetchTodaySteps, 30000);

        return () => {
            isMounted = false;
            if (intervalId) clearInterval(intervalId);
        };
    }, [token]);

    const fetchSteps = async (token, todayStart, todayEnd, weekStart) => {
        const today = await fetchStepCount(token, todayStart, todayEnd);
        const weekly = await fetchStepCount(token, weekStart, todayEnd);
        return { today, weekly };
    };

    const fetchCalories = async (token, todayStart, todayEnd, weekStart) => {
        const today = await fetchCalorieExpenditure(token, todayStart, todayEnd);
        const weekly = await fetchCalorieExpenditure(token, weekStart, todayEnd);
        return { today, weekly };
    };

    const fetchStepCount = async (token, startTime, endTime) => {
        const response = await fetch("https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${token}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                aggregateBy: [{
                    dataTypeName: "com.google.step_count.delta",
                    dataSourceId: "derived:com.google.step_count.delta:com.google.android.gms:merge_step_deltas"
                }],
                bucketByTime: { durationMillis: 60000 }, // 1-minute buckets
                startTimeMillis: startTime,
                endTimeMillis: endTime
            })
        });

        if (!response.ok) throw new Error('Failed to fetch steps');

        const data = await response.json();
        return calculateTotalSteps(data);
    };

    const fetchCalorieExpenditure = async (token, startTime, endTime) => {
        const response = await fetch("https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${token}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                aggregateBy: [{
                    dataTypeName: "com.google.calories.expended",
                    dataSourceId: "derived:com.google.calories.expended:com.google.android.gms:merge_calories_expended"
                }],
                bucketByTime: { durationMillis: 86400000 },
                startTimeMillis: startTime,
                endTimeMillis: endTime
            })
        });

        if (!response.ok) throw new Error('Failed to fetch calories');

        const data = await response.json();
        return calculateTotalCalories(data);
    };

    const calculateTotalSteps = (data) => {
        return data.bucket?.reduce((total, bucket) => {
            const points = bucket.dataset?.[0]?.point || [];
            return total + points.reduce((sum, point) => {
                return sum + (point.value?.[0]?.intVal || 0);
            }, 0);
        }, 0) || 0;
    };

    const calculateTotalCalories = (data) => {
        return data.bucket?.reduce((total, bucket) => {
            const points = bucket.dataset?.[0]?.point || [];
            return total + points.reduce((sum, point) => {
                return sum + (point.value?.[0]?.fpVal || 0);
            }, 0);
        }, 0) || 0;
    };

    return { 
        todaySteps, 
        weeklySteps, 
        todayCalories, 
        weeklyCalories, 
        isLoading, 
        error 
    };
};

export default useFitnessData;
