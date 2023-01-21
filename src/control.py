# %%
# the job list
import logging


class Job_list:
    """
    A list of jobs to be executed by the container program
    Currently this is expected to be a list of document ids
    to be processed by the container program. They will be
    treated as strings, but could be integers in some cases.

    Attributes:
        jobs (list): A list of jobs to be executed

    Methods:
        add(job): Adds a job to the list
        get_first_job(): Returns the first job in the list
        remove(job): Removes a job from the list
        __len__(): Returns the length of the list
        __str__(): Returns the list as a string
        __repr__(): Returns the list as a string
    """

    def __init__(self):
        """
        Initializes the job list.
        """
        self.jobs = []
        self.logger = logging.getLogger(__name__)

    def add(self, job: str):
        """
        Adds a job to the list
        Args:
            job (str): The job to be added

        Returns:
            None
        """
        self.jobs.append(job)

    def get_first_job(self):
        """
        Returns the first job in the list,
        and removes it from the list

        Returns:
            str: The first job in the list
        """
        return self.jobs.pop(0) if len(self.jobs) > 0 else None

    def remove(self, job: str):
        """
        Removes a job from the list
        Args:
            job (str): The job to be removed

        Returns:
            None
        """
        self.jobs.remove(job)

    def bulk_add(self, jobs: list):
        """
        Adds a list of jobs to the list
        Args:
            jobs (list): The jobs to be added

        Returns:
            None
        """
        self.jobs.extend(jobs)

    def bulk_remove(self, jobs: list):
        """
        Removes a list of jobs from the list
        Args:
            jobs (list): The jobs to be removed

        Returns:
            None
        """
        for job in jobs:
            self.remove(job)

    def clear(self):
        """
        Clears the list of jobs

        Returns:
            None
        """
        self.jobs = []

    def __len__(self):
        return len(self.jobs)

    def __str__(self):
        return str(self.jobs)

    def __repr__(self):
        return str(self.jobs)


# %%
