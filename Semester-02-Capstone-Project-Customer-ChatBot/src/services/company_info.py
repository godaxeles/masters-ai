"""Company information service."""

from typing import Dict

import structlog

from config import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)


class CompanyInfoService:
    """Service for managing company information."""

    def __init__(self):
        """Initialize company info service."""
        self.company_data = {
            'name': settings.company_name,
            'email': settings.company_email,
            'phone': settings.company_phone,
            'website': settings.company_website,
            'address': settings.company_address
        }

        logger.info("Company info service initialized", company=self.company_data['name'])

    def get_company_info(self) -> str:
        """Get formatted company information.

        Returns:
            Formatted company information string
        """
        info_parts = [
            f"**{self.company_data['name']}**",
            "",
            f"📧 Email: {self.company_data['email']}",
            f"📞 Phone: {self.company_data['phone']}",
            f"🌐 Website: {self.company_data['website']}",
            f"📍 Address: {self.company_data['address']}"
        ]

        return "\n".join(info_parts)

    def get_company_data(self) -> Dict[str, str]:
        """Get company data as dictionary.

        Returns:
            Dictionary with company information
        """
        return self.company_data.copy()

    def update_company_info(self, **kwargs) -> None:
        """Update company information.

        Args:
            **kwargs: Company information fields to update
        """
        for key, value in kwargs.items():
            if key in self.company_data:
                self.company_data[key] = value
                logger.info("Company info updated", field=key, value=value)

    def get_support_contact_info(self) -> str:
        """Get support-specific contact information.

        Returns:
            Support contact information
        """
        return (
            f"For additional support, you can reach us at:\n"
            f"Email: {self.company_data['email']}\n"
            f"Phone: {self.company_data['phone']}\n"
            f"Website: {self.company_data['website']}"
        )

