import { createBrowserRouter } from "react-router-dom";
import App from "../App";

import ChallengeList from "../components/ChallengeList";
import ChallengeForm from "../components/ChallengeForm";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
    //   {
    //     path: "/",
    //     element: <GoogleAuth />
    //   },
    //   {
    //     path: "/home",
    //     element: <Home />
    //   },
    //   Challenge Routes
      {
        path: "/challenges",
        element: <ChallengeList />
      },
      {
        path: "/challenges/new",
        element: <ChallengeForm />
      },
      {
        path: "/challenges/edit/:id",
        element: <ChallengeForm />
      }
    ]
  }
]);

export default router;

