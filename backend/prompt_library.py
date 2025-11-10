import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class PromptLibrary:
    """
    Manages saved prompts in a local JSON database
    """

    def __init__(self, db_path: str = "prompts.json"):
        self.db_path = Path(db_path)
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Create database file if it doesn't exist"""
        if not self.db_path.exists():
            self._write_db([])

    def _read_db(self) -> List[Dict]:
        """Read prompts from database"""
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []

    def _write_db(self, prompts: List[Dict]):
        """Write prompts to database"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)

    def save_prompt(self, prompt_data: Dict) -> Dict:
        """Save a new prompt"""
        prompts = self._read_db()

        new_prompt = {
            "id": str(datetime.now().timestamp()).replace('.', ''),
            "name": prompt_data.get("name", "Unnamed Prompt"),
            "prompt": prompt_data["prompt"],
            "thumbnail": prompt_data.get("thumbnail"),
            "tags": prompt_data.get("tags", []),
            "saved_at": datetime.now().isoformat(),
            "user_input": prompt_data.get("user_input", ""),
            "metadata": prompt_data.get("metadata", {})
        }

        prompts.insert(0, new_prompt)  # Add to beginning
        self._write_db(prompts)

        return new_prompt

    def get_all_prompts(self) -> List[Dict]:
        """Get all saved prompts"""
        return self._read_db()

    def get_prompt(self, prompt_id: str) -> Optional[Dict]:
        """Get a specific prompt by ID"""
        prompts = self._read_db()
        for prompt in prompts:
            if prompt["id"] == prompt_id:
                return prompt
        return None

    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt"""
        prompts = self._read_db()
        original_len = len(prompts)
        prompts = [p for p in prompts if p["id"] != prompt_id]

        if len(prompts) < original_len:
            self._write_db(prompts)
            return True
        return False

    def search_prompts(self, query: str) -> List[Dict]:
        """Search prompts by name, tags, or content"""
        prompts = self._read_db()
        query_lower = query.lower()

        results = []
        for prompt in prompts:
            if (query_lower in prompt["name"].lower() or
                query_lower in str(prompt.get("tags", [])).lower() or
                query_lower in prompt["prompt"].get("positive_prompt", "").lower()):
                results.append(prompt)

        return results

    def update_prompt(self, prompt_id: str, updates: Dict) -> Optional[Dict]:
        """Update a prompt"""
        prompts = self._read_db()

        for i, prompt in enumerate(prompts):
            if prompt["id"] == prompt_id:
                prompt.update(updates)
                prompt["updated_at"] = datetime.now().isoformat()
                self._write_db(prompts)
                return prompt

        return None
