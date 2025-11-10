import os
from pathlib import Path
from typing import Dict, Optional, List
import shutil
from datetime import datetime


class SettingsService:
    """
    Manages application settings including .env file operations
    """

    def __init__(self, env_path: str = ".env"):
        self.env_path = Path(env_path)
        self.backup_dir = Path(".env_backups")
        self.backup_dir.mkdir(exist_ok=True)

    def get_all_settings(self) -> Dict[str, str]:
        """
        Read all settings from .env file
        Returns dict with setting names as keys
        """
        settings = {}

        if not self.env_path.exists():
            return settings

        with open(self.env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Parse key=value
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    settings[key] = value

        return settings

    def mask_sensitive_value(self, key: str, value: str) -> str:
        """
        Mask sensitive values (API keys) for safe display
        Shows only last 4 characters
        """
        sensitive_keys = ['API_KEY', 'SECRET', 'PASSWORD', 'TOKEN']

        if any(sensitive in key.upper() for sensitive in sensitive_keys):
            if len(value) > 4:
                return '*' * (len(value) - 4) + value[-4:]
            return '*' * len(value)

        return value

    def get_safe_settings(self) -> Dict[str, str]:
        """
        Get all settings with sensitive values masked
        Safe for sending to frontend
        """
        settings = self.get_all_settings()
        safe_settings = {}

        for key, value in settings.items():
            safe_settings[key] = self.mask_sensitive_value(key, value)

        return safe_settings

    def backup_env(self) -> str:
        """
        Create a backup of the current .env file
        Returns the backup file path
        """
        if not self.env_path.exists():
            raise FileNotFoundError(".env file not found")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f".env.backup_{timestamp}"

        shutil.copy2(self.env_path, backup_path)

        # Keep only last 10 backups
        backups = sorted(self.backup_dir.glob(".env.backup_*"))
        if len(backups) > 10:
            for old_backup in backups[:-10]:
                old_backup.unlink()

        return str(backup_path)

    def update_settings(self, updates: Dict[str, str]) -> bool:
        """
        Update settings in .env file
        Creates backup before making changes

        Args:
            updates: Dict of setting_name -> new_value

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup first
            self.backup_env()

            # Read current .env file
            lines = []
            if self.env_path.exists():
                with open(self.env_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

            # Update or add settings
            updated_keys = set()
            new_lines = []

            for line in lines:
                stripped = line.strip()

                # Keep comments and empty lines as-is
                if not stripped or stripped.startswith('#'):
                    new_lines.append(line)
                    continue

                # Check if this line has a setting to update
                if '=' in stripped:
                    key = stripped.split('=', 1)[0].strip()

                    if key in updates:
                        # Update this setting
                        new_lines.append(f"{key}={updates[key]}\n")
                        updated_keys.add(key)
                    else:
                        # Keep original line
                        new_lines.append(line)
                else:
                    new_lines.append(line)

            # Add any new settings that weren't in the file
            for key, value in updates.items():
                if key not in updated_keys:
                    new_lines.append(f"{key}={value}\n")

            # Write updated .env file
            with open(self.env_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

            return True

        except Exception as e:
            print(f"Error updating settings: {e}")
            return False

    def validate_setting(self, key: str, value: str) -> Dict[str, any]:
        """
        Validate a single setting

        Returns:
            Dict with 'valid' (bool), 'errors' (list), 'warnings' (list)
        """
        errors = []
        warnings = []

        # Port number validation
        if 'PORT' in key.upper():
            try:
                port = int(value)
                if port < 1 or port > 65535:
                    errors.append(f"Port must be between 1 and 65535")
            except ValueError:
                errors.append(f"Port must be a number")

        # URL validation
        if 'URL' in key.upper() or 'HOST' in key.upper():
            if not value.startswith(('http://', 'https://')):
                warnings.append(f"URL should start with http:// or https://")

        # Boolean validation
        if value.lower() in ['true', 'false']:
            # Valid boolean
            pass
        elif key.upper() in ['USE_DUAL_LLM', 'OLLAMA_AUTO_CONFIGURE', 'DEBUG']:
            warnings.append(f"Boolean value should be 'true' or 'false'")

        # Number validation for generation defaults
        if key == 'DEFAULT_STEPS':
            try:
                steps = int(value)
                if steps < 1 or steps > 150:
                    errors.append("Steps must be between 1 and 150")
            except ValueError:
                errors.append("Steps must be a number")

        if key == 'DEFAULT_CFG_SCALE':
            try:
                cfg = float(value)
                if cfg < 1.0 or cfg > 30.0:
                    warnings.append("CFG scale typically between 1.0 and 30.0")
            except ValueError:
                errors.append("CFG scale must be a number")

        # Image dimensions
        if key in ['DEFAULT_WIDTH', 'DEFAULT_HEIGHT']:
            try:
                dim = int(value)
                if dim < 64 or dim > 2048:
                    warnings.append(f"Dimension should be between 64 and 2048")
                if dim % 8 != 0:
                    warnings.append(f"Dimension should be divisible by 8 for best results")
            except ValueError:
                errors.append(f"Dimension must be a number")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def validate_all_settings(self, settings: Dict[str, str]) -> Dict[str, any]:
        """
        Validate all settings at once

        Returns:
            Dict with 'valid' (bool), 'results' (dict of validation results per key)
        """
        results = {}
        all_valid = True

        for key, value in settings.items():
            result = self.validate_setting(key, value)
            results[key] = result
            if not result['valid']:
                all_valid = False

        return {
            'valid': all_valid,
            'results': results
        }

    def restore_from_backup(self, backup_path: str) -> bool:
        """
        Restore .env from a backup file
        """
        try:
            backup = Path(backup_path)
            if not backup.exists():
                return False

            shutil.copy2(backup, self.env_path)
            return True
        except Exception as e:
            print(f"Error restoring backup: {e}")
            return False

    def get_available_backups(self) -> List[Dict[str, str]]:
        """
        Get list of available backup files
        """
        backups = []
        for backup_file in sorted(self.backup_dir.glob(".env.backup_*"), reverse=True):
            backups.append({
                'path': str(backup_file),
                'name': backup_file.name,
                'timestamp': backup_file.stat().st_mtime
            })
        return backups
